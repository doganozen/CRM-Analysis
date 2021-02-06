
#pip install mysql-connector-python-rf
import datetime as dt
import pandas as pd
import pymysql
import pymysql.connections
import mysql.connector
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)

pd.set_option('display.max_columns', None)

##########################################
# From db
##########################################
# credentials.
creds = {'user': 'synan',
         'passwd': 'haydegidelum',
         'host': 'db.github.rocks',
         'port': 3306,
         'db': 'group5'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)
retail_mysql_df.info()

##########################################
# From csv
##########################################

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

##########################################
# Data Preperation
##########################################

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def crm_data_prep(dataframe):
    dataframe.dropna(axis=0, inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    return dataframe


check_df(df)
df_prep = crm_data_prep(df)
check_df(df_prep)

##########################################
# Creating RFM Segments
##########################################

def create_rfm(dataframe):
    # RFM METRIKLERININ HESAPLANMASI
    # Dikkat! RFM için frekanslar nunique.

    today_date = dt.datetime(2011, 12, 11)

    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})

    rfm.columns = ['recency', 'frequency', "monetary"]

    rfm = rfm[(rfm['monetary'] > 0)]


    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    # Monetary segment tanımlamada kullanılmadığı için işlemlere alınmadı.

    # SEGMENTLERIN ISIMLENDIRILMESI
    rfm['rfm_segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['rfm_segment'] = rfm['rfm_segment'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "rfm_segment"]]
    return rfm


rfm = create_rfm(df_prep)
rfm.head()

##########################################
# Calculated CLTV
##########################################

def create_cltv_c(dataframe):
    # avg_order_value
    dataframe['avg_order_value'] = dataframe['monetary'] / dataframe['frequency']

    # purchase_frequency
    dataframe["purchase_frequency"] = dataframe['frequency'] / dataframe.shape[0]

    # repeat rate & churn rate
    repeat_rate = dataframe[dataframe.frequency > 1].shape[0] / dataframe.shape[0]
    churn_rate = 1 - repeat_rate

    # profit_margin
    dataframe['profit_margin'] = dataframe['monetary'] * 0.05

    # Customer Value
    dataframe['cv'] = (dataframe['avg_order_value'] * dataframe["purchase_frequency"])

    # Customer Lifetime Value
    dataframe['cltv'] = (dataframe['cv'] / churn_rate) * dataframe['profit_margin']

    # minmaxscaler
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(dataframe[["cltv"]])
    dataframe["cltv_c"] = scaler.transform(dataframe[["cltv"]])

    dataframe["cltv_c_segment"] = pd.qcut(dataframe["cltv_c"], 3, labels=["C", "B", "A"])

    dataframe = dataframe[["recency", "frequency", "monetary", "rfm_segment",
                           "cltv_c", "cltv_c_segment"]]

    return dataframe


check_df(rfm)


rfm_cltv = create_cltv_c(rfm)
check_df(rfm_cltv)

rfm_cltv.tail()

##########################################
# Predicted CLTV
##########################################

def create_cltv_p(dataframe):
    today_date = dt.datetime(2011, 12, 11)

    ## recency kullanıcıya özel dinamik.
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max()-date.min()).days,
                                                                lambda date: (today_date - date.min()).days],
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    rfm.columns = rfm.columns.droplevel(0)

    ## recency_cltv_p
    rfm.columns = ['recency_cltv_p', 'T', 'frequency', 'monetary']

    ## basitleştirilmiş monetary_avg
    rfm["monetary"] = rfm["monetary"] / rfm["frequency"]

    rfm.rename(columns={"monetary": "monetary_avg"}, inplace=True)


    # BGNBD için WEEKLY RECENCY VE WEEKLY T'nin HESAPLANMASI
    ## recency_weekly_cltv_p
    rfm["recency_weekly_cltv_p"] = rfm["recency_cltv_p"] / 7
    rfm["T_weekly"] = rfm["T"] / 7



    # KONTROL
    rfm = rfm[rfm["monetary_avg"] > 0]

    ## recency filtre (daha saglıklı cltvp hesabı için)
    rfm = rfm[(rfm['frequency'] > 1)]

    rfm["frequency"] = rfm["frequency"].astype(int)

    # BGNBD
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfm['frequency'],
            rfm['recency_weekly_cltv_p'],
            rfm['T_weekly'])

    # exp_sales_1_month
    rfm["exp_sales_1_month"] = bgf.predict(4,
                                           rfm['frequency'],
                                           rfm['recency_weekly_cltv_p'],
                                           rfm['T_weekly'])
    # exp_sales_3_month
    rfm["exp_sales_3_month"] = bgf.predict(12,
                                           rfm['frequency'],
                                           rfm['recency_weekly_cltv_p'],
                                           rfm['T_weekly'])

    # expected_average_profit
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(rfm['frequency'], rfm['monetary_avg'])
    rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                             rfm['monetary_avg'])
    # 6 aylık cltv_p
    cltv = ggf.customer_lifetime_value(bgf,
                                       rfm['frequency'],
                                       rfm['recency_weekly_cltv_p'],
                                       rfm['T_weekly'],
                                       rfm['monetary_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)

    rfm["cltv_p"] = cltv

    # minmaxscaler
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(rfm[["cltv_p"]])
    rfm["cltv_p"] = scaler.transform(rfm[["cltv_p"]])

    # rfm.fillna(0, inplace=True)

    # cltv_p_segment
    rfm["cltv_p_segment"] = pd.qcut(rfm["cltv_p"], 3, labels=["C", "B", "A"])

    ## recency_cltv_p, recency_weekly_cltv_p
    rfm = rfm[["recency_cltv_p", "T", "monetary_avg", "recency_weekly_cltv_p", "T_weekly",
               "exp_sales_1_month", "exp_sales_3_month", "expected_average_profit",
               "cltv_p", "cltv_p_segment"]]


    return rfm

rfm_cltv_p = create_cltv_p(df_prep)
check_df(rfm_cltv_p)

crm_final = rfm_cltv.merge(rfm_cltv_p, on="Customer ID", how="left")
check_df(crm_final)

#cltv_p en yüksek 2., monetary_avg en yüksek 2.
crm_final.sort_values(by="cltv_p", ascending=False).head() #18102
crm_final.sort_values(by="monetary_avg", ascending=False).head() #12590
crm_final.loc[crm_final.index == 18102]
crm_final.loc[crm_final.index == 12590]

# avg monetary ye göre sıraladığımızda gelen 2. kullanıcı bizden alışveriş yaptuğında büyük miktarda para bırakmış
# ancak frekans ve recency değerleri kötü olduğu için hibernating grubuna girmiş. cltv_p sıralamasında 2. kişi çok yakın bir
# zamanda alışveriş yapmış yüksek bi frekansa sahip monetary ve diğer değerleri de gayet güzel bu kullanıcı
# champions sınıfında. zaten kullanıcının champions sınıfında olmaması bizi şaşırtırdı
#cltv_c aynı cltv_p farklı
crm_final.loc[crm_final.index == 12347.0]
crm_final.loc[crm_final.index == 12348.0]

##########################################
# Veri Tabanına Gönderme
##########################################
crm_final.head(10)

crm_final.index.name = "CustomerID"
crm_final.index = crm_final.index.astype(int)

crm_final.to_sql(name="crm_final_dogan_ozen",
                 con=conn,
                 if_exists='replace',
                 index=True,  # index var o da aşağıdaki
                 index_label="CustomerID")
## hatayı çözemediğim için veritabanına gönderme işlemini yapmamdım
##mysql-connector-python-rf  bu kütüphaneyi yüklerken bi hata veriyor