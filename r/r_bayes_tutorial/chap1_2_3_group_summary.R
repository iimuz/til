library("LearnBayes")

data(studentdata)

# データの確認
studentdata[1, ]

# 集計
attach(studentdata)
table(Drink)

# グラフ表示
# barplot(table(Drink), xlab="Drink", ylab="Count")

plot(1:2, 3:4)
