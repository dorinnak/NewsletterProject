import subprocess

def install(name):
    subprocess.call(['pip', 'install', name])

install("feedparser")
install("newspaper3k")
install("DateTime")
install("pandas")
install("regex")
install("nltk")
install("scikit-learn")
install("numpy")
install("matplotlib")
install("Flask")
install("urllib5")
install("Unidecode")