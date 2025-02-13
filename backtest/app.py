from flask import Flask, request, jsonify, render_template
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import matplotlib
from matplotlib.ticker import MultipleLocator
from datetime import datetime, timedelta
from matplotlib.ticker import MaxNLocator


# 设置 Matplotlib 以支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 省略掉一些无关紧要的警告
import warnings
warnings.filterwarnings('ignore')

def Average(data, n): #data为收盘价序列，n为时间窗口   同时，data为pd.Series，因此要使用df['close']来作为输入
    return pd.Series.rolling(data, n).mean()

app = Flask(__name__)

def fetch_stock_data(ts_token, ts_code, start_date, end_date):

    # 将日期字符串转换为日期对象
    date_format = "%Y-%m-%d"
    date_obj = datetime.strptime(start_date, date_format)
    # 计算新的日期
    new_start_date = date_obj - timedelta(days=60)

    new_start_date = new_start_date.strftime("%Y-%m-%d")
    #设置接口并读取数据
    ts.set_token(ts_token)
    pro = ts.pro_api()
    df = pro.daily(**{
        "ts_code": ts_code,  # 股票代码为“平安银行”
        "trade_date": "",
        "start_date": new_start_date.replace('-', ''),   # 由于回测日期从20230101开始，因此数据起始日期从20221101开始，方便获取均线数据
        "end_date": end_date.replace('-', ''),    # 回测截止日期
        "offset": "",
        "limit": ""
    }, fields=[
        "ts_code",
        "trade_date",
        "close"
    ])
    # 将数据全部倒叙
    df = df.iloc[::-1].reset_index(drop=True)
    #print(df)
    #重新设置索引
    df = df.set_index('trade_date') # 设置索引为日期

    #print(df)
    #df.to_excel('stock_data.xlsx')
    return df

def calculate_indicators(df, short_window, long_window, initial_money, buy_rate, cost_rate,start_date,end_date):
    #df['short_mavg'] = df['close'].rolling(window=short_window, min_periods=1).mean()
    #df['long_mavg'] = df['close'].rolling(window=long_window, min_periods=1).mean()
    #df['signal'] = np.where(df['short_mavg'] > df['long_mavg'], 1, -1)

    # 计算短均线
    df['short_mavg'] = Average(df['close'], short_window)
    # 计算长均线
    df['long_mavg'] = Average(df['close'], long_window)
    # 计算交易信号
    df['inf'] = 0.0 # 初始化signal列为0
    df['inf'] = np.where(df['short_mavg'] > df['long_mavg'], 1, 0) # 当短均线大于长均线时，signal为1，否则为0
    df['trade'] = df['inf'].diff() # 当signal发生变化时，trade发生变化,代表是否进行交易

    # 对数据进行截断，仅从start_date开始
    print('数据截断')
    df = df.loc[start_date.replace('-', ''):]
    print(df)
    
    df['stock_num'] = 0
    df['cash'] = initial_money
    df['total'] = df['cash'] + df['stock_num'] * df['close']

    # 进行交易
    for i in range(1, len(df)-1):
        if i>len(df)-2: # 如果i超出了df的长度，就跳出循环,结束策略回测
            break
        else :        # 如果i还在范围内，则继续进行回测
            if df['trade'][i] == 1:
                # 买入股票的数量（向下取整）
                df['stock_num'][i+1] = int(buy_rate*df['cash'][i+1] / df['close'][i+1])
                # 更新现金持有量
                print('购买股票')
                # 需要减去交易费率
                df['cash'][i + 1] = df['cash'][i] - df['stock_num'][i+1] * df['close'][i] * (1 + cost_rate)
                # 更新总资产价值
                df['total'][i + 1] = df['cash'][i + 1] + df['stock_num'][i+1] * df['close'][i+1]  
            elif df['trade'][i] == -1:
                # 清仓股票
                df['stock_num'][i+1] = 0   # 卖出股票，股票数量为0
                # 更新现金持有量，需要减去交易费率
                df['cash'][i + 1] = df['close'][i + 1] * df['stock_num'][i]*(1 - cost_rate) + df['cash'][i]   # 更新现金持有量
                df['total'][i + 1] = df['cash'][i + 1] + df['stock_num'][i+1] * df['close'][i + 1]  # 更新总资产价值
            elif df['trade'][i] == 0:
                # 没有交易信号，则保持股票的持仓不变
                df['stock_num'][i + 1] = df['stock_num'][i]
                # 现金持有量不变
                df['cash'][i + 1] = df['cash'][i]
                # 总资产随着股票价值而变化
                df['total'][i + 1] = df['cash'][i+1] + df['stock_num'][i+1]*df['close'][i+1]
    
    #print("策略回测完成")
    #print(df)

    # 计算每日收益率
    df['daily_return'] = df['total'].pct_change()

    # 计算累计收益率
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

    # 计算最大收益率
    df['cumulative_min'] = df['total'].cummin()
    df['max_return'] = (df['total'] - df['cumulative_min']) / df['cumulative_min']
    max_return = df['max_return'].max()


    # 计算累计收益率
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

    # 计算累计最大值
    df['cumulative_max'] = df['cumulative_return'].cummax()

    # 计算回撤
    df['drawdown'] = df['cumulative_return'] - df['cumulative_max']

    # 最大回撤
    max_drawdown = df['drawdown'].min()

    cum_return = df['cumulative_return'].iloc[-1]

    final_return = cum_return*initial_money

    #print(cum_return)
    #print(max_return)
    #print(max_drawdown)

    return df, cum_return, max_return, max_drawdown,final_return

def plot_chart(df):
    fig, ax = plt.subplots(figsize=(15, 8))
    # 绘制曲线，指定不同的颜色
    ax.plot(df.index, df['close'], label='收盘价', color='black')
    ax.plot(df.index, df['short_mavg'], label='12日均线', color='orange')
    ax.plot(df.index, df['long_mavg'], label='26日均线', color='blue')
    # 仅标记 trade 列中值为 1 和 -1 的点
    df_pos = df[df['trade'] == 1]
    df_neg = df[df['trade'] == -1]

    ax.scatter(df_pos.index, df_pos['short_mavg'], color='green', marker='x', label='买入点')
    ax.scatter(df_neg.index, df_neg['short_mavg'], color='red', marker='x', label='卖出点')
    # 索引间隔表示每个月
    ax.xaxis.set_major_locator(MultipleLocator(90))  # 每隔2个单位显示一个刻度
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return encoded_img

def plot_total(df):
    # 创建对象
    fig, ax = plt.subplots(figsize=(15, 8))

    # 绘制曲线，指定不同的颜色
    ax.plot(df.index, df['total'], label='总资产价值', color='black')

    # 仅标记 trade 列中值为 1 和 -1 的点
    df_pos = df[df['trade'] == 1]
    df_neg = df[df['trade'] == -1]

    ax.scatter(df_pos.index, df_pos['total'], color='green', marker='x', label='买入点')
    ax.scatter(df_neg.index, df_neg['total'], color='red', marker='x', label='卖出点')

    # 禁用科学计数法
    ax.ticklabel_format(style='plain', axis='y')

    # 标注 total 序列的最大值和最小值点
    max_idx = df['total'].idxmax()
    min_idx = df['total'].idxmin()
    ax.scatter(max_idx, df['total'].max(), color='blue', s=100, label='最大值')
    ax.scatter(min_idx, df['total'].min(), color='orange', s=100, label='最小值')

    # 设置 x 轴刻度间隔
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return encoded_img



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/backtest', methods=['POST'])
def backtest():
    data = request.json
    df = fetch_stock_data(data['ts_token'], data['ts_code'], data['start_date'], data['end_date'])
    df, cum_return, max_return, max_drawdown,final_return = calculate_indicators(
        df, data['short_window'], data['long_window'], data['initial_money'], data['buy_rate'], data['cost_rate'],data['start_date'],data['end_date'])
    img1 = plot_chart(df)
    img2 = plot_total(df)

    return jsonify({
        "cumulative_return": cum_return,
        "max_return": max_return,
        "max_drawdown": max_drawdown,
        "final_return":final_return,
        "img1": img1,
        "img2": img2
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
