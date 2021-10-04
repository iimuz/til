const path = require("path")
const HtmlWebpackPlugin = require("html-webpack-plugin")

module.exports = {
  entry: './src/main.ts',
  output: {
    path: path.resolve(__dirname, "dist"),
    publicPath: "dist",
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
      inject: "head",
      hash: true,
    }),
    new HtmlWebpackPlugin({
      template: path.resolve(__dirname, "src", "html", "sample.html"),
      filename: "html/sample.html",
      inject: "head",
      hash: true,
    })
  ],
  devServer: {
    static: "dist/html",
    open: false,
  }
};