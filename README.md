# bridge_test

使用Flas可进行后端开发，在网页前端有以下参数需要输入

TuShare Token：（请输入你本人的tushare的token）

股票代码：（请输入平安银行的股票代码 000001.SZ     想输入其他的代码也可以）

起始日期：（使用下拉列表选择2023-01-01）

结束日期：（使用下拉列表选择2023-12-31）

初始资金：（输入 1000000）

购买比例：（输入0.20）

短均线窗口：（输入12）

长均线窗口：（输入26）

费率：（输入0.0001）


→ 以上参数也可以修改


## 返回

页面在后端运行将会返回五项内容
1. 累计收益率
2. 最大收益率
3. 最大回撤

以及两张图像
4. 均线走势图
5. 总资产走势图


## 使用方法

1.使用vscode或pycharm等打开app.py后点击运行

2.点击链接http://127.0.0.1:5000 或 http://192.168.71.65:5000

3.等待加载一段时间后，网页弹出，在网页中输入参数后点击”运行回测“按钮  （有时可能加载较慢）

4.结果输出并可以查看

PS：原始代码为一个名为Bridge_TEST的jupyter文件，已经放入了github中，app.py为该ipynb文件移植到开发后端的代码



