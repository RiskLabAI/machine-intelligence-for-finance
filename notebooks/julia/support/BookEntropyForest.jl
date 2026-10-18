module BookEntropyForest

using ..BookForest: SplitStream, draw_position!

mutable struct EntropyNode
    feature::Int
    threshold::Float64
    left::Int
    right::Int
    count::Int
    mass::Float64
    negative_mass::Float64
    positive_mass::Float64
    impurity::Float64
end

struct EntropyTree
    nodes::Vector{EntropyNode}
    importance::Vector{Float64}
    in_bag::Vector{Int}
end

function exchange!(values,rows,a,b)
    values[a],values[b] = values[b],values[a]
    rows[a],rows[b] = rows[b],rows[a]
end

function sift_down!(values,rows,offset,root,n)
    while true
        child = 2root+1
        largest = root
        child<n && values[offset+largest]<values[offset+child] && (largest=child)
        child+1<n && values[offset+largest]<values[offset+child+1] && (largest=child+1)
        largest==root && return
        exchange!(values,rows,offset+root,offset+largest)
        root = largest
    end
end

function heap_sort!(values,rows,offset,n)
    for start in fld(n-2,2):-1:0
        sift_down!(values,rows,offset,start,n)
    end
    for last in n-1:-1:1
        exchange!(values,rows,offset,offset+last)
        sift_down!(values,rows,offset,0,last)
    end
end

function sort_pairs!(values,rows,offset=1,n=length(rows),depth=2floor(Int,log2(max(n,1))))
    while n>15
        if depth<=0
            heap_sort!(values,rows,offset,n)
            return
        end
        depth -= 1
        a,b,c = values[offset],values[offset+fld(n,2)],values[offset+n-1]
        pivot = a<b ? (b<c ? b : (a<c ? c : a)) : (b<c ? (a<c ? a : c) : b)
        index,left,right = 0,0,n
        while index<right
            if values[offset+index]<pivot
                exchange!(values,rows,offset+index,offset+left)
                index += 1
                left += 1
            elseif values[offset+index]>pivot
                right -= 1
                exchange!(values,rows,offset+index,offset+right)
            else
                index += 1
            end
        end
        sort_pairs!(values,rows,offset,left,depth)
        offset += right
        n -= right
    end
    for index in offset+1:offset+n-1
        value,row = values[index],rows[index]
        position = index
        while position>offset && values[position-1]>value
            values[position],rows[position] = values[position-1],rows[position-1]
            position -= 1
        end
        values[position],rows[position] = value,row
    end
end

function entropy(mass,counts)
    value = 0.
    for count in counts
        if count>0
            probability = count/mass
            value -= probability*(log(probability)/log(2.))
        end
    end
    return value
end

function totals(rows,labels,weights)
    mass = 0.
    counts = zeros(2)
    for row in rows
        weight = weights[row]
        counts[labels[row]+1] += weight
        mass += weight
    end
    return mass,counts
end

function update_left!(left,position,boundary,rows,labels,weights,mass,counts,left_mass)
    if boundary-position<=length(rows)-boundary
        for index in position+1:boundary
            row = rows[index]
            left[labels[row]+1] += weights[row]
            left_mass += weights[row]
        end
    else
        left .= counts
        left_mass = mass
        for index in length(rows):-1:boundary+1
            row = rows[index]
            left[labels[row]+1] -= weights[row]
            left_mass -= weights[row]
        end
    end
    return left_mass
end

function best_partition!(stream,features,labels,weights,rows,known,max_features,min_leaf,mass,counts)
    available = length(stream.order)
    discovered,visited,drawn = 0,0,0
    constants = known
    best_feature,best_threshold,best_proxy,best_position = 0,NaN,-Inf,0
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
        values = features[rows,feature]
        sort_pairs!(values,rows)
        if values[end]<=values[1]+Float32(1e-7)
            stream.order[position],stream.order[constants+1] = stream.order[constants+1],stream.order[position]
            discovered += 1
            constants += 1
            continue
        end
        stream.order[available],stream.order[position] = stream.order[position],stream.order[available]
        available -= 1
        left = zeros(2)
        left_mass,last_position,boundary = 0.,0,0
        while boundary<length(rows)
            boundary += 1
            while boundary<length(rows) && values[boundary+1]<=values[boundary]+Float32(1e-7)
                boundary += 1
            end
            boundary==length(rows) && continue
            boundary>=min_leaf && length(rows)-boundary>=min_leaf || continue
            left_mass = update_left!(left,last_position,boundary,rows,labels,weights,mass,counts,left_mass)
            last_position = boundary
            right_mass = mass-left_mass
            left_mass>=0 && right_mass>=0 || continue
            proxy = -right_mass*entropy(right_mass,counts.-left)-left_mass*entropy(left_mass,left)
            if proxy>best_proxy
                best_feature = feature
                best_threshold = Float64(values[boundary])/2+Float64(values[boundary+1])/2
                best_proxy,best_position = proxy,boundary
            end
        end
    end
    stream.order[1:known] = stream.constants[1:known]
    stream.constants[known+1:constants] = stream.order[known+1:constants]
    best_feature==0 && return (feature=0,threshold=NaN,constants=constants,position=0,left_impurity=NaN,right_impurity=NaN,left_mass=NaN,right_mass=NaN)
    position,last = 1,length(rows)
    while position<=last
        if features[rows[position],best_feature]<=best_threshold
            position += 1
        else
            rows[position],rows[last] = rows[last],rows[position]
            last -= 1
        end
    end
    @assert last==best_position
    left = zeros(2)
    left_mass = update_left!(left,0,best_position,rows,labels,weights,mass,counts,0.)
    right_mass = mass-left_mass
    return (feature=best_feature,threshold=best_threshold,constants=constants,position=best_position,left_impurity=entropy(left_mass,left),right_impurity=entropy(right_mass,counts.-left),left_mass=left_mass,right_mass=right_mass)
end

"""
Fit finite binary entropy CART with supplied positive sample weights and split
seed. Weighted class counts follow sequential forward/reverse accumulation;
paired Float32 introsorting preserves public split and near-tie ordering.
"""
function fit_tree(x,y,weights,split_seed;max_depth=typemax(Int),min_leaf=1,max_features=1)
    size(x,1)==length(y)==length(weights) || throw(DimensionMismatch("tree inputs must align"))
    all(isfinite,x) && all(in((0,1)),y) || throw(ArgumentError("finite features and binary labels are required"))
    all(value->isfinite(value) && value>=0,weights) && any(>(0),weights) || throw(ArgumentError("nonnegative weights with positive mass are required"))
    max_depth>=1 && min_leaf>=1 && 1<=max_features<=size(x,2) || throw(ArgumentError("invalid tree growth settings"))
    features,labels,weights = Float32.(x),Int.(y),Float64.(weights)
    stream = SplitStream(UInt32(split_seed),collect(1:size(x,2)),zeros(Int,size(x,2)))
    nodes = EntropyNode[]
    in_bag = findall(>(0),weights)
    root_mass,root_counts = totals(in_bag,labels,weights)
    function grow(rows,depth,known,impurity)
        mass,counts = totals(rows,labels,weights)
        node = EntropyNode(0,NaN,0,0,length(rows),mass,counts[1],counts[2],impurity)
        push!(nodes,node)
        index = length(nodes)
        if depth>=max_depth || length(rows)<max(2,2min_leaf) || impurity<=eps(Float64)
            return index
        end
        split = best_partition!(stream,features,labels,weights,rows,known,max_features,min_leaf,mass,counts)
        split.feature==0 && return index
        improvement = (mass/root_mass)*(impurity-(split.right_mass/mass)*split.right_impurity-(split.left_mass/mass)*split.left_impurity)
        improvement+eps(Float64)<0 && return index
        node.feature,node.threshold = split.feature,split.threshold
        node.left = grow(rows[1:split.position],depth+1,split.constants,split.left_impurity)
        node.right = grow(rows[split.position+1:end],depth+1,split.constants,split.right_impurity)
        return index
    end
    grow(copy(in_bag),0,0,entropy(root_mass,root_counts))
    importance = zeros(size(x,2))
    for node in nodes
        if node.feature>0
            left,right = nodes[node.left],nodes[node.right]
            importance[node.feature] += node.mass*node.impurity-left.mass*left.impurity-right.mass*right.impurity
        end
    end
    importance ./= root_mass
    normalizer = sum(importance)
    normalizer>0 && (importance./=normalizer)
    return EntropyTree(nodes,importance,in_bag)
end

function predict_proba(tree::EntropyTree,x)
    result = zeros(size(x,1),2)
    for row in axes(x,1)
        index = 1
        while tree.nodes[index].feature>0
            node = tree.nodes[index]
            index = Float32(x[row,node.feature])<=node.threshold ? node.left : node.right
        end
        node = tree.nodes[index]
        result[row,:] = [node.negative_mass/node.mass,node.positive_mass/node.mass]
    end
    return result
end

predict(tree::EntropyTree,x) = [argmax(row)-1 for row in eachrow(predict_proba(tree,x))]

export EntropyTree,fit_tree,predict_proba,predict

end
