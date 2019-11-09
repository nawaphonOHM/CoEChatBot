from flask_restful import request
import flask
import flask_restful
import flask_cors
import json
import os

from modeling.run_model import response


class ChatWithBot(flask_restful.Resource):
    def post(self) -> list:
        data_message = json.loads(request.data)
        response_json = {}

        try:
            if not data_message["sender"] or len(data_message) == 0:
                raise AttributeError()
            response_message = response(data_message["msg"], data_message["state"])
            state = None
            
            if response_message[0]["intention_set"] == False:
                state = data_message["state"]
            else:
                state = response_message[0]["intention_set"]

            response_json["msg"] = response_message[0]["response_sentence"]
            response_json["state"] = state
            response_json["sender"] = False
            response_json["input_type"] = response_message[1]
            response_json["input_type_chooise"] = response_message[2]
            response_json["response_chooise"] = response_message[3]

        except Exception:
            response_json["msg"] = None
            response_json["state"] = None
            response_json["sender"] = None
        
        return flask.Response(\
                json.dumps(response_json), 
                content_type="application/json; charset=UTF-8"
            )

app = flask.Flask(__name__)
api = flask_restful.Api(app)
flask_cors.CORS(app)
api.add_resource(ChatWithBot, "/chatwithbot")
os.environ["PYTHONPATH"] = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    app.run(port=1996)  