const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  entry: "./src/main.tsx",
  output: {
    path: `${__dirname}/build`,
    filename: "main.js",
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
      },
      {
        test: /\.css?$/,
        use: [
          "style-loader",
          "css-modules-typescript-loader",
          {
            loader: "css-loader",
            options: {
              importLoaders: 1,
              modules: true,
            }
          }
        ]
      }
    ]
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: './src/index.html',
      filename: 'index.html',
    }),
  ],
  resolve: {
    extensions: [".ts", ".tsx", ".js", ".json", ".css"]
  },
  target: ["web", "es5"],
}
