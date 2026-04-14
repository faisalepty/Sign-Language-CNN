# base/consumers.py
import json
import base64
import io
from channels.generic.websocket import AsyncWebsocketConsumer
from PIL import Image
import torch
from .ml.predictor import predictor

class PredictConsumer(AsyncWebsocketConsumer):

    # client connected
    async def connect(self):
        await self.accept()
        print("Client connected ✅")

    # client disconnected
    async def disconnect(self, close_code):
        print("Client disconnected")

    # client sent a frame
    async def receive(self, text_data):
        try:
            data = json.loads(text_data)

            # decode base64 image from browser
            image_data   = data['image'].split(',')[1]  # remove "data:image/jpeg;base64,"
            image_bytes  = base64.b64decode(image_data)

            # run prediction
            result = predictor.predict(image_bytes)

            # send result back immediately
            await self.send(text_data=json.dumps({
                'label'     : result['label'],
                'confidence': result['confidence'],
                'success'   : True
            }))

        except Exception as e:
            await self.send(text_data=json.dumps({
                'success': False,
                'error'  : str(e)
            }))