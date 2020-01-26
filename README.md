+++++++++++++++++++++++++++++
Automated Newsletter Creation 
+++++++++++++++++++++++++++++

1 - Included files
===========================

Create_Newsletter.py	--- run this code and the HTML newsletter will be created

newsletter.py ----------- code for newsletter creation

package_installer.py ---- code for automated installation of requiered Python packages for newsletter.py

URLs_finance.json	------- list of all URLs to be scraped by newsletter.py (links can be changed/replaced anytime)

folder: logos	----------- logos used by newsletter.py for the HTML file

folder: templates	------- HTML template used by newsletter.py

! Be sure to have all the files in one and the same folder when running.


2 - Creating the newsletter
===========================

Run "Create_Newsletter.py"

To create a newsletter, the most simple way is to run "Create_Newsletter.py"
Using "URLs_finance.json", this file automatically outputs the newsletter as an HTML file.

- In CMD in your Python installation folder: python Create_Newsletter.py
- In Python: import Create_Newsletter


3 - Required Python Version and Packages
===========================

newsletter.py runs on Python 3.7.3 and makes use of other Python packages which need to be installed before running.

Installation of all required packages can be automated with the included "package_installer.py" file.

- In CMD in your Python installation folder: python package_installer.py
- In Python: import package_installer

This is a list of all packages used and it's corresponding version:

pip install feedparser / 5.2.1

pip install newspaper3k /  0.2.8

pip install DateTime / 4.3

pip install pandas / 0.24.2

pip install regex / 2.2.1

pip install nltk / 3.4

pip install scikit-learn / 0.20.3

pip install numpy / 1.16.2

pip install matplotlib / 3.0.3

pip install Flask / 1.0.2

pip install urllib5 / 5.0.0

pip install Unidecode / 1.1.1



