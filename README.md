# Enabling Customizable Recommender Systems through Explainable Machine Learning
Authored by Alexander Einarsson, Andong Li Zhao, Sergio Servantez, Jakub Wasylkowski


In this repo we provide an end-to-end recommender system prototype where nontechnical users can easily understand what data has been used in making recommendations, and allow users to customize these inputs.

## Project Deployment

We tried to make interacting with the system as easy as possible. We provide two methods for deploying the project:

### 1. Hosted Site
We have hosted the recommender system on a server which you can visit [here](https://hcml-project.herokuapp.com/). Please note that we are hosting the website on a free service so the server sometimes goes down or runs into timeout errors. If you run into any issues, contact Sergio to restart the server. 

### 2. Local Deployment
If you weren't able to access the system through the link, we have also set up the app so that it is easy to run locally. Download this repo and open terminal to run the following commands:

`cd [project directory]`

`python3 -m venv venv` 

`. venv/bin/activate`

`pip install -r requirements.txt`

`python wsgi.py`

Then simply cut and paste the generated local host URL into a web browser and you are ready to go!
