<!-- PROJECT SHIELDS -->
[![PRETTIER](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://gitter.im/jlongster/prettie)
[![LICENSE](https://img.shields.io/github/license/arshadkazmi42/awesome-github-init.svg)](https://github.com/arshadkazmi42/awesome-github-init/LICENSE)
[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/paulo-mota-955218a2/)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/paulowiz/disaster_response_pipeline=%23E71A18&title_bg=%23555555&icon=dependabot.svg&icon_color=%23E7E7E7&title=views&edge_flat=false)](https://hits.seeyoufarm.com)
<!-- PROJECT SHIELDS -->



<!-- PROJECT -->
<p align="center">
  <h3 align="center"> 
   Disaster Response Pipeline 
  </h3> 
  <p align="center">
    <img alt="Disaster" src="https://github.com/paulowiz/assets/blob/main/disaster_project.gif">
    <br />
    <br />
    <a href="https://github.com/lucioerlan/Soap-Automation/issues">Report Bug</a>
    ·
    <a href="https://github.com/lucioerlan/Soap-Automation/issues">Request Feature</a>
  </p>
</p>



<!-- ABOUT THE PROJECT -->
## 🤔 Introduction
Disaster messages have been used for any emergencial situations in the world, people who needs help or needs to report disaster situations related about enviroment, security, wether, etc. The project consist in a webapp where the user can input a new message and get classification results in several categories (Floating, Military, Earthquake, etc.), classified by machine learning model that I have trained.



<br /> 

---
<!-- SETUP -->

## 🔥 Setup

#### Copy or rename the file

```
$ pip install requirements.txt
```

<br />

---


<!-- INSTALLATION -->

## 🔨 Installation and Running

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_project.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_project.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

<br />

---

#### Or Run Docker 🐳
```
$ docker build . --tag img_disaster_project:latest &&
$ docker rm -f disaster_project &&
$ docker run -d  -p 80:80 --name disaster_project img_disaster_project:latest &&
```
<br />

---

<!-- File Overview -->

## 📚 Project files overview

- app
  - template
    - master.html # main page of web app
    - go.html # classification result page of web app
   - run.py # Flask file that runs app
- data
  - disaster_categories.csv # data to process
  - disaster_messages.csv # data to process
  - process_data.py # clean, process and save data for training the machine learning model
  - disaster_project.db # database to save clean data to
- models
  - train_classifier.py # train model
  - classifier.pkl # saved model 
- README.md
<!-- LICENSE -->

## 🔓 License

This project lives under MIT License. See LICENSE for more details. © - [Paulo Mota](https://www.linkedin.com/in/paulo-mota-955218a2/)

<br />



<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
<!-- MARKDOWN LINKS & IMAGES -->


