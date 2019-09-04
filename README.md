# HerokuPlotlyDashboard
Step 1. Create a new folder for your project:
```ruby
$ mkdir dash_app_example
$ cd dash_app_example
```

Step 2. Initialize the folder with git and a virtualenv

$ git init        # initializes an empty git repo
$ virtualenv venv # creates a virtualenv called "venv"
$ source venv/bin/activate # uses the virtualenv

virtualenv creates a fresh Python instance. You will need to reinstall your app's dependencies with this virtualenv:

$ pip install dash
$ pip install plotly

You will also need a new dependency, gunicorn, for deploying the app:

$ pip install gunicorn

Step 3. Initialize the folder with a sample app (app.py), a .gitignore file, requirements.txt, and a Procfile for deployment

Create the following files in your project folder:

app.py (include plotly dash code)

.gitignore
    venv
    *.pyc
    .DS_Store
    .env

Procfile
    web: gunicorn app:server
   
(Note that app refers to the filename app.py. server refers to the variable server inside that file).

requirements.txt (Describes your Python dependencies. You can fill this file in automatically with)
$ pip freeze > requirements.txt

4. Initialize Heroku, add files to Git, and deploy

$ heroku create my-dash-app # change my-dash-app to a unique name
$ git add . # add all files to git
$ git commit -m 'Initial app boilerplate'
$ git push heroku master # deploy code to heroku
$ heroku ps:scale web=1  # run the app with a 1 heroku "dyno"

You should be able to view your app at https://my-dash-app.herokuapp.com (changing my-dash-app to the name of your app).

5. Update the code and redeploy

When you modify app.py with your own code, you will need to add the changes to git and push those changes to heroku.

$ git status # view the changes
$ git add .  # add all the changes
$ git commit -m 'a description of the changes'
$ git push heroku master
