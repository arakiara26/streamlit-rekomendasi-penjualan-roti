import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori
import ast

# load dataset
df = pd.read_csv("bread_basket.csv")
df['date_time'] = pd.to_datetime(df['date_time'], format= "%d-%m-%Y %H:%M")

df["month"] = df['date_time'].dt.month
df["day"] = df['date_time'].dt.weekday

# Penggantian nama bulan dan hari
df["month"].replace(
    [i for i in range(1, 13)], 
    ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"], 
    inplace=True
)
df["day"].replace(
    [i for i in range(7)], 
    ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"], 
    inplace=True
)

# Misalkan period_day dan weekday_weekend didefinisikan berdasarkan waktu dan hari
df["period_day"] = np.where(df['date_time'].dt.hour < 12, 'pagi', 
                            np.where(df['date_time'].dt.hour < 18, 'siang', 
                                     np.where(df['date_time'].dt.hour < 21, 'sore', 'malam')))
df["weekday_weekend"] = np.where(df['day'].isin(["Sabtu", "Minggu"]), 'weekend', 'weekday')

# Tampilkan judul di Streamlit
st.title("Rekomendasi Penjualan Roti dengan Algoritma Apriori")

def get_data(period_day='', weekday_weekend='', month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data["period_day"].str.contains(period_day, case=False, na=False)) &
        (data["weekday_weekend"].str.contains(weekday_weekend, case=False, na=False)) &
        (data["month"].str.contains(month, case=False, na=False)) &
        (data["day"].str.contains(day, case=False, na=False))
    ]
    return filtered if not filtered.empty else "No Result!"

def user_input_features():
    item = st.selectbox("Menu", ['Bread', 'Scandinavian', 'Hot chocolate', 'Jam', 'Cookies', 'Muffin', 'Coffee', 'Pastry', 'Medialuna', 'Tea', 'Tartine', 'Basket', 'Mineral Water', 'Farm House', 'Fudge', 'Juice', 'Ella Kitchen Pouches', 'Victorian Sponge', 'Frittata', 'Hearty & Seasonal', 'Soup', 'Pick and Mix Bowls', 'Smoothies', 'Cake', 'Mighty Protein', 'Chicken sand', 'Coke', 'My-5 Fruit Shoot', 'Focaccia', 'Sandwich', 'Alfajores', 'Eggs', 'Brownie', 'Dulce de Leche', 'Honey', 'The BART', 'Granola', 'Fairy Doors', 'Empanadas', 'Keeping It Local', 'Art Tray', 'Bowl Nic Pitt', 'Bread Pudding', 'Adjustment', 'Truffles', 'Chimichurri Oil', 'Bacon', 'Spread', 'Kids Biskuit', 'Siblings', 'Caramel bites', 'Jammie Dodgers', 'Tiffin', 'Olum & polenta', 'Polenta', 'The Nomad', 'Hack the stack', 'Bakewell', 'Lemon and coconut', 'Toast', 'Scone', 'Crepes', 'Vegan mincepie', 'Bare Popcorn', 'Muesli', 'Crisps', 'Pintxos', 'Gingerbread syrup', 'Panettone', 'Brioche and salami', 'Afternoon with the baker', 'Salad', 'Chicken Stew', 'Spanish Brunch', 'Raspberry shortbread sandwich', 'Extra Salami or Feta', 'Duck Egg', 'Baguette', "Valentine's card", 'Tshirt', 'Vegan Feast', 'Postcard', 'Nomad bag', 'Chocolates', 'Coffee granules', 'Drink chocolate spoons', 'Christmas common', 'Argentina Night', 'Half slice Monster', 'Gift voucher', 'Cherry me Dried fruit', 'Mortimer', 'Raw bars', 'Tacos/Fajita'])
    period_day = st.selectbox('Waktu Pembelian', ['Pagi', 'Siang', 'Malam'])
    weekday_weekend = st.selectbox('Weekday / Weekend', ['Weekend', 'Weekday'])
    month= st.select_slider("Bulan", ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"])
    day = st.select_slider("Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"], value="Sabtu")

    return period_day.lower(), weekday_weekend.lower(), month, day, item

period_day, weekday_weekend, month, day, item = user_input_features()

data = get_data(period_day, weekday_weekend, month, day)

def encode(x):
    if x <=0:
        return 0
    elif x >= 1:
        return 1

if type(data) != str:
    item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="count")
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.01
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1

    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    filtered_data = data[data["antecedents"] == item_antecedents]
    if not filtered_data.empty:
        return list(filtered_data.iloc[0, :])
    else:
        return ["No matching antecedents found!", ""]

if type(data) != str:
    st.markdown("Hasil Rekomendasi : ")
    result = return_item_df(item)
    st.success(f"Jika konsumen membeli **{item}**, maka konsumen membeli **{result[1]}** secara bersamaan")
else:
    st.error("Tidak Ada Hasil!")