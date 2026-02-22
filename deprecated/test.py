import openai
import requests
import base64
from PIL import Image
from io import BytesIO
import json

import os

os.environ["OPENAI_API_KEY"] = ""

client = openai.OpenAI()

image_front_train_path = "./train_front_0.png"
image_side_train_path = "./train_side_0.png"
image_front_test_path = "./test_front_0.png"
image_side_test_path = "./test_side_0.png"

prompt = f"""
The robot will be performing the task of putting the pink cube on the blue cube according to a policy trained on provided dataset.

The robot will be using two cameras to observe the environment and make decisions based on the visual input. The whole environment will follow a certain protocol to ensure the overall consistency of the settings.

Under this protocol, the basic elements of the environment will be the same. However, the lighting, the pose of the cameras may vary on a small scale due to unavoidable errors.

Now I want you to consider the images of the initial setting of any rollout and analyze its difficulty compared to the training dataset.

Although I will not be abled to provide you with the complete training dataset, I can offer you some relative difference between the test case and the training data in the <rel> and </rel> tags below.

<rel>

L2 distance of the nearest training data: 5.0cm

relative camera pose: 0.1cm, 0.1cm, 0.1rad

</rel>

You will analyze the difficulty of the task based on the image provided. The difficulty should be assessed based on how well the robot can perform the task given the visual input from the camera.

Based on the image of the initial setting of any rollout, you must analyze its difficulty compared to the training dataset and give a specific difficulty value.

First analyze the task in the <think> and </think> tags below:

<think>

[Write your detailed analysis here]
</think>

Then list out all the factors that may influence the outcome of the process and classify them into certain levels of difficulty.

Finally provide a difficulty value from 1 to 10 between <difficulty> and </difficulty> tags where:
- 10 means the test case is very hard and the model is expected to achieve a success rate of 0% on it
- 8-9 means the test case is rather hard and the model is expected to achieve a success rate of 10%-20% on it
- 5-7 means the test case is moderate in difficulty and the model is expected to achieve a success rate of 30%-50% on it
- 2-4 means the test case is rather easy and the model is expected to achieve a success rate of 60%-80% on it
- 1 means the test case is very easy and the model is expected to achieve a success rate of 90%-100% on it

<difficulty>

[Write your levels of difficulty here]

</difficulty>
"""

desc_front_train_img = f"""
This image below shows the front camera view in one of the training data.
"""

desc_side_train_img = f"""
This image below shows the side camera view in one of the training data.
"""

desc_front_test_img = f"""
This image below shows the front camera view of the test case that needs to be evaluated.
"""

desc_side_test_img = f"""
This image below shows the side camera view of the test case that needs to be evaluated.
"""

with open(image_front_train_path, "rb") as image_file:
    b64_front_train_image = base64.b64encode(image_file.read()).decode("utf-8")
with open(image_side_train_path, "rb") as image_file:
    b64_side_train_image = base64.b64encode(image_file.read()).decode("utf-8")

with open(image_front_test_path, "rb") as image_file:
    b64_front_test_image = base64.b64encode(image_file.read()).decode("utf-8")
with open(image_side_test_path, "rb") as image_file:
    b64_side_test_image = base64.b64encode(image_file.read()).decode("utf-8")

response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_text", "text": desc_front_train_img},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64_front_train_image}",
                },
                {"type": "input_text", "text": desc_side_train_img},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64_side_train_image}",
                },
                {"type": "input_text", "text": desc_front_test_img},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64_front_test_image}",
                },
                {"type": "input_text", "text": desc_side_test_img},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64_side_test_image}",
                },
            ],
        }
    ],
)

print(response.output_text)
