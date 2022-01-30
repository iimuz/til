import Pkg
Pkg.add("Weave")

using Weave

convert_doc("src/plot_sin.jl", "converted/plot_sin.ipynb")
