from django.shortcuts import render,redirect
from django.conf import settings
from django.contrib import messages


import inspect
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import json
from yahoo_finance import Share

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.stats import norm

import matplotlib.pyplot as plt

from PIL import Image
import plotnine

from .models import Stock
from .forms import StockForm


def search_stock(base_url, stock_ticker):
    try:
        token = settings.TWELVEDATA_API
        url_hl = 'https://api.twelvedata.com/time_series?symbol='
        url = base_url + stock_ticker + '&apikey=' + token
        turl = url_hl + stock_ticker + '&interval=1day&outputsize=1&apikey=' + token
        data = requests.get(url)
        data_hl = requests.get(turl)

        if data.status_code == 200 and data_hl.status_code == 200:
            data = json.loads(data.content)
            data_hl = json.loads(data_hl.content)

        else:
            data = {'Error' : 'There was a problem with your provided ticker symbol. Please try again'}
    except Exception as e:
        data = {'Error':'There has been some connection error. Please try again later.'}
    return data, data_hl

def search_stock_batch(base_url, stock_tickers):
    data_list = []
    tl_list = []
    url_hl = 'https://api.twelvedata.com/time_series?symbol='
    try:
        token = settings.TWELVEDATA_API
        for ticker in stock_tickers:
            url = base_url + ticker + '&apikey=' + token
            turl = url_hl + ticker + '&interval=1day&outputsize=1&apikey=' + token
            data = requests.get(url)
            data_hl = requests.get(turl)

            if data.status_code == 200:
                data = json.loads(data.content)
                data_hl = json.loads(data_hl.content)
                data_list.append(data)
                tl_list.append(data_hl)
            else:
                data = {'Error' : 'There has been an unexpected issues. Please try again'}
    except Exception as e:
        data = {'Error':'There has been some connection error. Please try again later.'}
    return data_list, tl_list

def check_valid_stock_ticker(stock_ticker):
    base_url = 'https://api.twelvedata.com/price?symbol='
    stock = search_stock(base_url, stock_ticker)
    if 'Error' not in stock:
        return True
    return False

def check_stock_ticker_existed(stock_ticker):
    try:
        stock = Stock.objects.get(ticker=stock_ticker)
        if stock:
            return True
    except Exception:
        return False

def home(request):
    if request.method == 'POST':
        stock_ticker = request.POST['stock_ticker']
        base_url ='https://api.twelvedata.com/price?symbol='
        stocks, stocks_hl = search_stock(base_url, stock_ticker)
        vl = {
                'symbol': stocks_hl['meta']['symbol'],
                'price' : stocks['price'],
                'open' : stocks_hl['values'][0]['open'],
                'high' : stocks_hl['values'][0]['high'],
                'low' : stocks_hl['values'][0]['low'],
                'close' : stocks_hl['values'][0]['close'],

            }
        return render(request, 'quotes/home.html', {'stocks':vl})
    return render(request, 'quotes/home.html')

def about(request):
    return render(request, 'quotes/about.html')

def portfolio(request):
    if request.method == 'POST':
        ticker = request.POST['ticker']
        if ticker:
            form = StockForm(request.POST or None)

            if form.is_valid():
                if check_stock_ticker_existed(ticker):
                    messages.warning(request, f'{ticker} is already existed in Portfolio.')
                    return redirect('portfolio')

                if check_valid_stock_ticker(ticker):
                    #add stock                    
                    form.save()
                    messages.success(request, f'{ticker} has been added successfully.')
                    return redirect('portfolio')

        messages.warning(request, 'Please enter a valid ticker name.')
        return redirect('portfolio')
    else:
        stockdata = Stock.objects.all()
        stkdata = []
        print(stockdata)
        if stockdata:
            ticker_list = [stock.ticker for stock in stockdata]
            ticker_list = list(set(ticker_list))
            
            tickers = ticker_list
            base_url = 'https://api.twelvedata.com/price?symbol='
            stockdata, ts_data = search_stock_batch(base_url, tickers)
            print(stockdata)
            print(ts_data)
            for (stock, tl) in zip(stockdata, ts_data):
                if stock:
                    vl = {
                        'symbol': tl['meta']['symbol'],
                        'price' : stock['price'],
                        'open' : tl['values'][0]['open'],
                        'high' : tl['values'][0]['high'],
                        'low' : tl['values'][0]['low'],
                        'close' : tl['values'][0]['close'],

                    }
                stkdata.append(vl);
        else:
            messages.info(request, 'Currently, there are no stocks in your portfolio!')
        if stkdata:
            return render(request, 'quotes/portfolio.html', {'stkdata':stkdata})
        else:
            return render(request, 'quotes/portfolio.html', {'stkdata':stockdata})



def risk_analysis(request):


    # Stock List provided by user 
    # ----------------------------------------------------------------------
    stockList = ['AAPL']
    # ----------------------------------------------------------------------


    ret_data = pd.read_csv(filepath_or_buffer="./FinalReturnData.csv")
    ret_data = ret_data[stockList]
    mean = ret_data.apply(func=np.mean, axis=0)
    std = ret_data.apply(func=np.std, axis=0)
    ret_data -= mean
    ret_data /= std

    class BidirectionalGenerativeAdversarialNetworkDiscriminator(tf.keras.Model):
        def __init__(self, num_hidden):
            super().__init__()

            args, _, _, values = inspect.getargvalues(frame=inspect.currentframe())
            values.pop("self")

            for arg, val in values.items():
                setattr(self, arg, val)

            self.concat = tf.keras.layers.Concatenate(axis=-1)
            self.feature_extractor = tf.keras.Sequential(
                layers=[
                    tf.keras.layers.Dense(
                        units=self.num_hidden,
                        activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                    ),
                ]
            )
            self.dropout = tf.keras.layers.Dropout(rate=0.5)
            self.discriminator = tf.keras.layers.Dense(
                units=1,
                activation="sigmoid",
            )

        def call(self, x, z):
            features = self.concat([x, z])
            features = self.feature_extractor(features)
            features = self.dropout(features)

            return self.discriminator(features)


    class BidirectionalGenerativeAdversarialNetworkGenerator(tf.keras.Model):
        def __init__(self, num_hidden, num_inputs):
            super().__init__()

            args, _, _, values = inspect.getargvalues(frame=inspect.currentframe())
            values.pop("self")

            for arg, val in values.items():
                setattr(self, arg, val)

            self.generator = tf.keras.Sequential(
                layers=[
                    tf.keras.layers.Dense(
                        units=self.num_hidden,
                        activation="elu",
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(
                        units=self.num_hidden,
                        activation="elu",
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(
                        units=self.num_inputs,
                        activation="linear",
                    ),
                ]
            )

        def call(self, z):
            return self.generator(z)


    class BidirectionalGenerativeAdversarialNetworkEncoder(tf.keras.Model):
        def __init__(self, num_hidden, num_encoding):
            super().__init__()

            args, _, _, values = inspect.getargvalues(frame=inspect.currentframe())
            values.pop("self")

            for arg, val in values.items():
                setattr(self, arg, val)

            self.encoder = tf.keras.Sequential(
                layers=[
                    tf.keras.layers.Dense(
                        units=self.num_hidden,
                        activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(
                        units=self.num_hidden,
                        activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                    ),
                    tf.keras.layers.Dense(
                        units=self.num_encoding,
                        activation="tanh",
                    ),
                ]
            )

        def call(self, x):
            return self.encoder(x)
        
    num_inputs = ret_data.shape[1]
    num_hidden = 200
    num_encoding = 100
    num_epochs = 4000
    batch_size = 100
    generator = BidirectionalGenerativeAdversarialNetworkGenerator(
        num_hidden=num_hidden, num_inputs=num_inputs
    )
    discriminator = BidirectionalGenerativeAdversarialNetworkDiscriminator(
        num_hidden=num_hidden
    )
    encoder = BidirectionalGenerativeAdversarialNetworkEncoder(
        num_hidden=num_hidden, num_encoding=num_encoding
    )

    ds = (
        tf.data.Dataset.from_tensor_slices(tensors=ret_data)
        .shuffle(buffer_size=ret_data.shape[0] * 2, reshuffle_each_iteration=True)
        .batch(batch_size=batch_size, drop_remainder=False)
    )

    reconstruction_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )
    optimizer_disc = tf.keras.optimizers.RMSprop(
        learning_rate=2e-4, decay=1e-8, clipvalue=1.0
    )
    optimizer_enc_gen = tf.keras.optimizers.RMSprop(
        learning_rate=4e-4, decay=1e-8, clipvalue=1.0
    )
    disc_loss_metric = tf.keras.metrics.Mean(name="train_disc_loss")
    enc_loss_metric = tf.keras.metrics.Mean(name="train_enc_loss")
    gen_loss_metric = tf.keras.metrics.Mean(name="train_gen_loss")


    @tf.function
    def train_step(x, real, z, fake):
        with tf.GradientTape() as dis_tape, tf.GradientTape(
            persistent=True
        ) as enc_gen_tape:
            enc = encoder(x=x, training=True)
            gen = generator(z=z, training=True)
            disc_loss_real = reconstruction_loss(
                y_true=real,
                y_pred=discriminator(
                    x=x,
                    z=enc,
                    training=True,
                ),
            )
            disc_loss_fake = reconstruction_loss(
                y_true=fake,
                y_pred=discriminator(
                    x=gen,
                    z=z,
                    training=True,
                ),
            )
            disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
            enc = encoder(x=x, training=True)
            gen = generator(z=z, training=True)
            enc_loss = reconstruction_loss(
                y_true=fake,
                y_pred=discriminator(
                    x=x,
                    z=enc,
                    training=True,
                ),
            )
            gen_loss = reconstruction_loss(
                y_true=real,
                y_pred=discriminator(
                    x=gen,
                    z=z,
                    training=True,
                ),
            )
        gradients_disc = dis_tape.gradient(
            target=disc_loss,
            sources=discriminator.trainable_variables,
        )
        optimizer_disc.apply_gradients(
            grads_and_vars=zip(
                gradients_disc,
                discriminator.trainable_variables,
            )
        )
        disc_loss_metric(disc_loss)
        gradients_enc = enc_gen_tape.gradient(
            target=enc_loss, sources=encoder.trainable_variables
        )
        optimizer_enc_gen.apply_gradients(
            grads_and_vars=zip(
                gradients_enc,
                encoder.trainable_variables,
            )
        )
        enc_loss_metric(enc_loss)
        gradients_gen = enc_gen_tape.gradient(
            target=gen_loss, sources=generator.trainable_variables)
       
        optimizer_enc_gen.apply_gradients(
            grads_and_vars=zip(
                gradients_gen,
                generator.trainable_variables,
            )
        )
        gen_loss_metric(gen_loss)


    for epoch in range(num_epochs):
        disc_loss_metric.reset_states()
        enc_loss_metric.reset_states()
        gen_loss_metric.reset_states()

        for x in ds:
            train_step(
                x=x,
                real=np.ones(shape=(x.shape[0], 1)),
                z=np.random.uniform(low=-1.0, high=1.0, size=(x.shape[0], num_encoding)),
                fake=np.zeros(shape=(x.shape[0], 1)),
            )

        if ((epoch + 1) % 1000) == 0:
            print("Epoch:", epoch + 1)
            print("Discriminator loss:", disc_loss_metric.result())
            print("Encoder loss:", enc_loss_metric.result())
            print("Generator loss:", gen_loss_metric.result())

    num_sim = 1000
    with tf.device(device_name="/CPU:0"):
        x_mean = [
            np.average(
                a=(
                    generator(
                        z=np.array(
                            object=[
                                np.random.uniform(low=-1.0, high=1.0, size=(num_encoding))
                            ]
                        )
                    )[0]
                    * std
                )
                + mean
            )
            for i in range(num_sim)
        ]

    act_mean = [
        np.average(a=(ret_data.iloc[i] * std) + mean) for i in range(ret_data.shape[0])
    ]

    plotnine.options.figure_size = (12, 9)
    plot = (
        plotnine.ggplot(
            mapping=pd.melt(
                frame=pd.concat(
                    objs=[
                        pd.DataFrame(
                            data=x_mean, columns=["BiGAN Portfolio Returns Distribution"]
                        ),
                        pd.DataFrame(
                            data=act_mean, columns=["Actual Portfolio Returns Distribution"]
                        ),
                    ],
                    axis=1,
                ).reset_index(drop=True)
            ).dropna(axis="index")
        )
        + plotnine.geom_density(
            mapping=plotnine.aes(
                x="value",
                fill="factor(variable)",
            ),
            alpha=0.5,
            color="black",
        )
        + plotnine.geom_point(
            mapping=plotnine.aes(x="value", y=0, fill="factor(variable)"),
            alpha=0.5,
            color="black",
        )
        + plotnine.xlab(xlab="Portfolio returns")
        + plotnine.ylab(ylab="Density")
        + plotnine.ggtitle(
            title="Trained Bidirectional Generative Adversarial Network (BiGAN) Portfolio Returns"
        )
        + plotnine.theme_matplotlib()
    )
    plot.save(filename="trained_bigan_sampler.png")
    print(
        "The VaR at 1% estimate given by the BiGAN: {}%".format(
            100 * np.percentile(a=x_mean, axis=0, q=1)
        )
    )  
    print(
        "The VaR at 5% estimate given by the BiGAN: {}%".format(
            100 * np.percentile(a=x_mean, axis=0, q=5)
        )
    )        
    print(
        "The VaR at 10% estimate given by the BiGAN: {}%".format(
            100 * np.percentile(a=x_mean, axis=0, q=10)
        )
    )        



    return render(request, 'quotes/risk_analysis.html')

def delete_stock(request, stock_symbol):
    stock = Stock.objects.get(ticker=stock_symbol)
    stock.delete()

    messages.success(request, f'{stock.ticker} has been deleted successfully.')
    return redirect('portfolio')