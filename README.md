+++++++++++++++++++++++++++++
Automated Newsletter Creation 
+++++++++++++++++++++++++++++

1 - Included files
===========================

Create_Newsletter.py	- run this code and all the packages will be installed and the HTML newsletter will be created
newsletter.py 		- code for newsletter creation
package_installer.py	- code for automated installation of requiered Python packages for newsletter.py

URLs_finance.json	- list of all URLs to be scraped by newsletter.py (links can be changed/replaced anytime)

folder: logos		- logos used by newsletter.py for the HTML file
folder: templates	- HTML template used by newsletter.py

! Be sure to have all the files in one and the same folder when running.


2 - Creating the newsletter
===========================

Run "Create_Newsletter.py"

To create a newsletter, the most simple way is to run "Create_Newsletter.py"
Using "URLs_finance.json", this file automatically outputs the newsletter as an HTML file.

- In CMD in your Python installation folder: python Create_Newsletter.py
- In Python: import Create_Newsletter


3 - Required Python Packages
===========================

newsletter.py makes use of other Python packages which need to be installed before running:

pip install feedparser
pip install newspaper3k 
pip install DateTime 
pip install pandas 
pip install regex 
pip install nltk 
pip install scikit-learn 
pip install numpy 
pip install matplotlib 
pip install Flask 
pip install urllib5 
pip install Unidecode 

Installation of all requiered packages can be automated with the included "package_installer.py" file.

- In CMD in your Python installation folder: python package_installer.py
- In Python: import package_installer
