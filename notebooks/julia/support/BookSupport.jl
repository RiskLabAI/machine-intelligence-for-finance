module BookSupport

using CSV, DataFrames, Dates, Plots, SHA, Statistics, Printf, TOML

const palette = (background="#111827", axes="#111827", text="#E5E7EB",
    cyan="#38BDF8", amber="#FBBF24", green="#34D399", rose="#FB7185", grid="#374151")

function configure_plots!()
    theme(:dark)
    default(background_color=palette.background, background_color_inside=palette.axes,
        foreground_color=palette.text, gridcolor=palette.grid, gridalpha=.5,
        fontfamily="sans-serif", linewidth=2, legendfontsize=8, guidefontsize=10,
        titlefontsize=11, tickfontsize=8, margin=5Plots.mm, dpi=150)
end

book_root() = normpath(joinpath(@__DIR__, "..", "..", ".."))

function output_paths(subject)
    root = joinpath(book_root(), "outputs", "julia")
    paths = (; figures=joinpath(root, "figures", subject), tables=joinpath(root, "tables", subject), figure_data=joinpath(root, "figure_data", subject))
    foreach(mkpath, paths)
    return paths
end

function save_figure(figure, paths, name)
    for extension in ("png", "pdf")
        savefig(figure, joinpath(paths.figures, name*"."*extension))
    end
    display(figure)
    return nothing
end

save_data(table, paths, name) = CSV.write(joinpath(paths.figure_data, name*".csv"), table)

_latex(value) = replace(string(value), '\\'=>raw"\textbackslash{}", '_'=>raw"\_", '%'=>raw"\%", '&'=>raw"\&", '#'=>raw"\#")
_number(value, digits) = value isa AbstractFloat ? (isfinite(value) ? @sprintf("%.*g", digits, abs(value)<5e-13 ? 0. : value) : string(value)) : string(value)

function save_table(table, paths, name; caption="", label="", digits=6)
    CSV.write(joinpath(paths.tables, name*".csv"), table)
    open(joinpath(paths.tables, name*".tex"), "w") do stream
        println(stream, raw"\begin{table}[tbp]\centering")
        !isempty(caption) && println(stream, "\\caption{", _latex(caption), "}")
        !isempty(label) && println(stream, "\\label{", label, "}")
        println(stream, "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{", repeat("l", ncol(table)), "}")
        println(stream, join(_latex.(names(table)), " & "), raw" \\\hline")
        for row in eachrow(table)
            println(stream, join([_latex(_number(value, digits)) for value in row], " & "), raw" \\")
        end
        println(stream, raw"\end{tabular}\end{adjustbox}\end{table}")
    end
    display(table)
    return nothing
end

function dataset_path(name)
    root = book_root()
    manifest = TOML.parsefile(joinpath(@__DIR__, "..", "data", "inputs.toml"))["datasets"][name]
    original = joinpath(root, manifest["original"])
    bytes2hex(sha256(read(original))) == manifest["original_sha256"] || error("The approved source data hash differs: $name")
    path = joinpath(root, manifest["native_input"])
    bytes2hex(sha256(read(path))) == manifest["native_sha256"] || error("The native input hash differs: $name")
    return path
end

function load_binance()
    path = dataset_path("binance_btcusdt")
    columns = ["Open time", "Open", "High", "Low", "close", "volume", "Close time", "quote_volume", "trades", "Buyer base volume", "Buyer quote volume", "Unused"]
    frame = CSV.read(path, DataFrame; header=columns)
    frame.time = DateTime(1970, 1, 1).+Millisecond.(frame[!, "Open time"])
    sort!(frame, :time)
    unique!(frame, :time)
    return frame
end

minimum_variance_weights(covariance) = (solution=covariance\ones(size(covariance, 1)); solution./sum(solution))

export palette, configure_plots!, book_root, output_paths, save_figure,
    save_table, save_data, dataset_path, load_binance, minimum_variance_weights

end
