import Pkg
Pkg.add("Weave")

using Weave

convert_doc("src/plot_cos.ipynb", "converted/plot_cos.jl")
