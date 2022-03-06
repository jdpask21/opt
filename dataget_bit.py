#蝋燭足データを取得する関数　引数：はじめの日付、終わりの日付、通貨ペア、ろうそくタイプ　返り値：データフレーム

def hourly_data(start,end,pair,candle_type):

    #ライブラリ
    import python_bitbankcc
    import datetime
    import pandas
    from tqdm import tqdm

    #パブリックAPIのオブジェクトを取得
    pub = python_bitbankcc.public()

    #引数を日付データに変換
    start_date = datetime.datetime.strptime(start,"%Y%m%d")
    end_date = datetime.datetime.strptime(end,"%Y%m%d")

    #日付の引き算
    span = end_date - start_date

    #データを入れる配列を定義しておく
    ohlcv_data = []


    #1時間ごとに時間足データを取得し、結合していく
    for counter in tqdm(range(span.days + 1)):

        #日付の計算
        the_day = start_date + datetime.timedelta(days = counter)

        #データが欠損している部分は無視する
        try:
            #パブリックAPIのインスタンス化
            value = pub.get_candlestick(pair, candle_type, the_day.strftime("%Y%m%d"))

            #データ部分の抽出
            ohlcv = value["candlestick"][0]['ohlcv']

            #結合
            ohlcv_data.extend(ohlcv)

        except:
            pass


    #データフレームに変換
    col = ["Open","High","Low","Close","Volume","Unix Time"]
    df_sum = pandas.DataFrame(ohlcv_data, columns = col)

    return df_sum

start = "20210101"
end = "20210201"
pair = "btc_jpy"
candle_type = "1hour"

df = hourly_data(start,end,pair,candle_type)
df.to_csv("data_bit_1y1h.csv")
#print(df)
