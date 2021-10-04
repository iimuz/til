const path = require("path")
const HtmlWebpackPlugin = require("html-webpack-plugin")

module.exports = {
  entry: {
    "index": "./src/main.ts",
    "sample": "./src/sample.ts",
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    publicPath: "dist",
    filename: "[name].js",
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
      },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: path.resolve(__dirname, "src", "html", "index.html"),
      filename: "html/index.html",
      chunks: ["index"],
      inject: "head",
      hash: true,
    }),
    new HtmlWebpackPlugin({
      template: path.resolve(__dirname, "src", "html", "sample.html"),
      filename: "html/sample.html",
      chunks: ["sample"],
      inject: "head",
      hash: true,
    })
  ],
  devServer: {
    static: "dist/html",
    open: false,
  }
};