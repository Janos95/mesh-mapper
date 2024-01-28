import openai
import base64
import requests
import os
import csv


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_description(api_key, image_path, side):
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"What’s in this image taken from the {side}? Focus on the object itself and not how it is rendered or triangulated."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response_json = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()

    if 'choices' in response_json and response_json['choices']:
        return response_json['choices'][0]['message']['content']
    else:
        return "No response content found."

def get_summary(api_key, descriptions):
    prompt = "The following descriptions describe the same object rendered from different views. Give a single brief description of the object. The descriptions are:\n" + "\n".join(descriptions)
    completion = openai.chat.completions.create(
      model="gpt-4",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
      ]
    )
    return completion.choices[0].message.content


def process_images_in_folders(folder_path, output_csv):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")

    image_sides = ['x_minus', 'x_plus', 'y_minus', 'y_plus']

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Directory', 'Description'])

        for subdir in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir)
            if os.path.isdir(subdir_path):
                descriptions = []
                for side in image_sides:
                    image_path = os.path.join(subdir_path, f'{side}.jpg')
                    if os.path.exists(image_path):
                        description = get_image_description(api_key, image_path, side)
                        descriptions.append(description)

                if descriptions:
                    summary = get_summary(api_key, descriptions)
                    print(f"Summary for {subdir}: {summary}")
                    csv_writer.writerow([subdir, summary])


# Example usage
folder_path = 'images'
output_csv = 'image_summaries.csv'
process_images_in_folders(folder_path, output_csv)
