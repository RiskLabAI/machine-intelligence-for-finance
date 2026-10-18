module BookForest

using Statistics, LinearAlgebra

mutable struct ClassificationNode
    feature::Int
    threshold::Float64
    left::Int
    right::Int
    count::Int
    mass::Float64
    positive_mass::Float64
    impurity::Float64
end

struct ClassificationTree
    nodes::Vector{ClassificationNode}
    importance::Vector{Float64}
    in_bag::Vector{Int}
end

struct ClassificationForest{S}
    specification::S
    trees::Vector{ClassificationTree}
end

mutable struct SplitStream
    state::UInt32
    order::Vector{Int}
    constants::Vector{Int}
end

function draw_position!(stream,lower,upper)
    state = stream.state==0 ? UInt32(1) : stream.state
    state ⊻= state<<13
    state ⊻= state>>17
    state ⊻= state<<5
    stream.state = state
    return lower+Int((state & UInt32(0x7fffffff))%UInt32(upper-lower))
end

gini(mass,positive) = 1-((mass-positive)^2+positive^2)/mass^2

function best_partition!(stream,features,labels,weights,rows,known,max_features,min_leaf)
    available = length(stream.order)
    discovered,visited,drawn = 0,0,0
    constants = known
    best_feature,best_threshold,best_proxy = 0,NaN,-Inf
    mass = sum(weights[rows])
    positive = sum(weights[rows].*labels[rows])
    while available>constants && (visited<max_features || visited<=discovered+drawn)
        visited += 1
        position = draw_position!(stream,drawn,available-discovered)+1
        if position<=known
            stream.order[drawn+1],stream.order[position] = stream.order[position],stream.order[drawn+1]
            drawn += 1
            continue
        end
        position += discovered
        feature = stream.order[position]
        sorted_rows = sort(rows;by=row->features[row,feature])
        values = features[sorted_rows,feature]
        if values[end]<=values[1]+Float32(1e-7)
            stream.order[position],stream.order[constants+1] = stream.order[constants+1],stream.order[position]
            discovered += 1
            constants += 1
            continue
        end
        stream.order[available],stream.order[position] = stream.order[position],stream.order[available]
        available -= 1
        left_mass,left_positive = 0.,0.
        for boundary in 1:length(sorted_rows)-1
            row = sorted_rows[boundary]
            left_mass += weights[row]
            left_positive += weights[row]*labels[row]
            values[boundary+1]<=values[boundary]+Float32(1e-7) && continue
            boundary>=min_leaf && length(rows)-boundary>=min_leaf || continue
            right_mass,right_positive = mass-left_mass,positive-left_positive
            left_mass>0 && right_mass>0 || continue
            proxy = -left_mass*gini(left_mass,left_positive)-right_mass*gini(right_mass,right_positive)
            if proxy>best_proxy
                best_feature = feature
                best_threshold = Float64(values[boundary])/2+Float64(values[boundary+1])/2
                best_proxy = proxy
            end
        end
    end
    stream.order[1:known] = stream.constants[1:known]
    stream.constants[known+1:constants] = stream.order[known+1:constants]
    return best_feature,best_threshold,constants
end

"""
Fit the book's finite binary Gini CART with supplied bootstrap multiplicities.
Leaf-size limits count distinct in-bag observations; class proportions and
impurity gains use multiplicities. Split evaluation and prediction use Float32
features, midpoint thresholds, strict first-maximum ties, the public near-constant
threshold and a supplied XorShift split stream. Nodes are stored in depth-first
order and feature importances normalize the total weighted impurity reduction.
"""
function fit_tree(x,y,weights,split_seed;max_depth=typemax(Int),min_leaf=1,max_features=size(x,2))
    size(x,1)==length(y)==length(weights) || throw(DimensionMismatch("tree inputs must align"))
    all(isfinite,x) && all(in((0,1)),y) || throw(ArgumentError("finite features and binary labels are required"))
    all(value->isfinite(value) && value>=0,weights) && sum(weights)>0 || throw(ArgumentError("nonnegative weights with positive mass are required"))
    max_depth>=1 && min_leaf>=1 && 1<=max_features<=size(x,2) || throw(ArgumentError("invalid tree growth settings"))
    features,labels,multiplicities = Float32.(x),Int.(y),Float64.(weights)
    stream = SplitStream(UInt32(split_seed),collect(1:size(x,2)),zeros(Int,size(x,2)))
    nodes = ClassificationNode[]
    importance = zeros(size(x,2))
    in_bag = findall(>(0),multiplicities)
    function grow(rows,depth,known)
        mass = sum(multiplicities[rows])
        positive = sum(multiplicities[rows].*labels[rows])
        impurity = gini(mass,positive)
        node = ClassificationNode(0,NaN,0,0,length(rows),mass,positive,impurity)
        push!(nodes,node)
        index = length(nodes)
        if depth>=max_depth || length(rows)<max(2,2min_leaf) || impurity<=eps(Float64)
            return index
        end
        feature,threshold,constants = best_partition!(stream,features,labels,multiplicities,rows,known,max_features,min_leaf)
        feature==0 && return index
        left_rows = filter(row->features[row,feature]<=threshold,rows)
        right_rows = filter(row->features[row,feature]>threshold,rows)
        left_mass = sum(multiplicities[left_rows])
        right_mass = mass-left_mass
        left_positive = sum(multiplicities[left_rows].*labels[left_rows])
        right_positive = positive-left_positive
        improvement = (mass/sum(multiplicities))*(impurity-(left_mass/mass)*gini(left_mass,left_positive)-(right_mass/mass)*gini(right_mass,right_positive))
        improvement+eps(Float64)<0 && return index
        node.feature,node.threshold = feature,threshold
        node.left = grow(left_rows,depth+1,constants)
        node.right = grow(right_rows,depth+1,constants)
        importance[feature] += mass*impurity-left_mass*gini(left_mass,left_positive)-right_mass*gini(right_mass,right_positive)
        return index
    end
    grow(in_bag,0,0)
    total = sum(importance)
    total>0 && (importance./=total)
    return ClassificationTree(nodes,importance,in_bag)
end

function predict_proba(tree::ClassificationTree,x)
    result = zeros(size(x,1),2)
    for row in axes(x,1)
        index = 1
        while tree.nodes[index].feature>0
            node = tree.nodes[index]
            index = Float32(x[row,node.feature])<=node.threshold ? node.left : node.right
        end
        node = tree.nodes[index]
        result[row,:] = [(node.mass-node.positive_mass)/node.mass,node.positive_mass/node.mass]
    end
    return result
end

predict(tree::ClassificationTree,x) = Int.(predict_proba(tree,x)[:,2].>.5)

"""Fit the specified binary Gini ensemble from fixed public bootstrap primitives."""
function fit_forest(specification,x,y,counts,split_seeds)
    size(counts)==(size(x,1),specification.trees) || throw(DimensionMismatch("bootstrap counts must align with observations and trees"))
    length(split_seeds)==specification.trees || throw(DimensionMismatch("one split seed is required per tree"))
    all(sum(counts;dims=1).==size(x,1)) || throw(ArgumentError("each bootstrap must contain the original observation count"))
    trees = [fit_tree(x,y,counts[:,index],split_seeds[index];max_depth=specification.depth,min_leaf=specification.leaf,max_features=specification.features==0 ? size(x,2) : specification.features) for index in 1:specification.trees]
    return ClassificationForest(specification,trees)
end

function predict_proba(forest::ClassificationForest,x)
    probabilities = zeros(size(x,1),2)
    for tree in forest.trees
        probabilities .+= predict_proba(tree,x)
    end
    return probabilities./length(forest.trees)
end

predict(forest::ClassificationForest,x) = Int.(predict_proba(forest,x)[:,2].>.5)
tree_importances(forest::ClassificationForest) = reduce(vcat,permutedims.(getfield.(forest.trees,:importance)))

function tree_records(forest::ClassificationForest)
    return [(features=[node.feature for node in tree.nodes if node.feature>0],thresholds=[node.threshold for node in tree.nodes if node.feature>0],bootstrap_indices=tree.in_bag) for tree in forest.trees]
end

export ClassificationTree,ClassificationForest,fit_tree,fit_forest,predict_proba,predict,tree_importances,tree_records

end
