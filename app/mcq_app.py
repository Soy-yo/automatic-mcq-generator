import os
import re

from flask import Flask, render_template, request, jsonify, make_response
from transformers import AutoTokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv


from mcq import (
    MultipleChoiceQuestionGenerator,
    QuestionGenerator,
    AnswerGenerator,
    DistractorGenerator
)

load_dotenv('app/.flaskenv')
app = Flask(__name__)

DEFAULT_MODEL_NAME = os.getenv('MODEL')

AG_MODEL_NAME = os.getenv('AG_MODEL', DEFAULT_MODEL_NAME)
QG_MODEL_NAME = os.getenv('QG_MODEL', DEFAULT_MODEL_NAME)
DG_MODEL_NAME = os.getenv('DG_MODEL', DEFAULT_MODEL_NAME)

BLANK_LINE_PATTERN = re.compile(r'\n\s*\n')

TOKENIZER = AutoTokenizer.from_pretrained('t5-small')

if len({AG_MODEL_NAME, QG_MODEL_NAME, DG_MODEL_NAME}) == 1:
    AG_MODEL = T5ForConditionalGeneration.from_pretrained(AG_MODEL_NAME)
    QG_MODEL = AG_MODEL
    DG_MODEL = AG_MODEL
else:
    AG_MODEL = T5ForConditionalGeneration.from_pretrained(AG_MODEL_NAME)
    QG_MODEL = T5ForConditionalGeneration.from_pretrained(QG_MODEL_NAME)
    DG_MODEL = T5ForConditionalGeneration.from_pretrained(DG_MODEL_NAME)


@app.route('/')
def index():
    return render_template('template.html')


@app.route('/documents/', methods=['POST'])
def post_documents():
    data = request.json
    text = data.get('text')
    kwargs = data.get('arguments', {})

    documents = BLANK_LINE_PATTERN.split(text)

    ag = AnswerGenerator(TOKENIZER, AG_MODEL)
    qg = QuestionGenerator(TOKENIZER, QG_MODEL)
    dg = DistractorGenerator(TOKENIZER, DG_MODEL)
    generator = MultipleChoiceQuestionGenerator(ag, qg, dg).fit(documents)

    results = generator.generate(**kwargs)

    response = make_response(
        jsonify([result.dict() for result in results])
    )
    response.headers['Content-Type'] = 'application/json'
    return response
