# Send SMS messages of current models stats and forecast for given Stock Ticker (SPY, AAPL, TSLA)
import smtplib

carriers = {
    'att': '@mms.att.net',
    'tmobile': '@tmomail.net',
    'verizon': '@vtext.com',
    'sprint': '@page.nextel.com'
}


def send(msg):
    # phone number censored for privacy :)
    to_number = '******8077{}'.format(carriers['verizon'])
    auth = ("stockforecastbot@gmail.com", "hgexfeyjgcoucitr")

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(auth[0], auth[1])

    server.sendmail(auth[0], to_number, msg)
