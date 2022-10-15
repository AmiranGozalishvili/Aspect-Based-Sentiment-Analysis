from fastapi import FastAPI

from commands import commands
from opinionparse import opinion_parser
from predictpolarity import dependency_matching, polarity

app = FastAPI()
run_all = commands()



@app.get('/')
def get_root():
    return {'message': 'Welcome to ASBA API'}

# 1 take text and apply (progress_apply(opinion_parser))  Review text to opinion
# 2 take opinion text and apply (progress_apply(polarity))  opinion to polarity
# 3 take opinion and apply (progress_apply(dependency_matching)) opinion to descriptors


@app.get('/ASBA')
def textanalyze(text):
    Opinion_text = opinion_parser(text)
    Polarity = polarity(text)
    Descriptors = dependency_matching(text)
    return ("Opinion Text", Opinion_text), ("Polarity", Polarity), ("Descriptor", Descriptors)
