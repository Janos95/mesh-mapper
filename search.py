from openai import OpenAI
import pandas as pd
import numpy as np
import scipy.spatial.distance
import polyscope as ps
import polyscope.imgui as psim
import trimesh
import os

client = OpenAI()

def find_file_by_name(root_dir, name):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[0] == name:
                return os.path.join(dirpath, filename)
    return None
def cosine_similarity(vec_a, vec_b):
    return 1 - scipy.spatial.distance.cosine(vec_a, vec_b)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def search_embedding(query, df):
    query_embedding = get_embedding(query)
    df['similarity'] = df['ada_embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    closest_match = df.sort_values('similarity', ascending=False).iloc[0]
    return closest_match

df = pd.read_csv('embedded_descriptions.csv')
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)

#query = "looking for an octopus model"
#closest_match = search_embedding(query, df)
#print(closest_match['Directory'])

prompt = ''
def callback():
    global prompt

    _, prompt = psim.InputText("Search Box", prompt)

    if psim.Button("Search"):
        closest_match = search_embedding(prompt, df)
        print(closest_match['Directory'])
        mesh_name = closest_match['Directory']
        mesh_path = find_file_by_name('meshes', mesh_name)
        mesh = trimesh.load(mesh_path)
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        ps_mesh = ps.register_surface_mesh('result', verts, faces)
        ps_mesh.set_color([0.5, 0.5, 0.5])  # Set color to grey

ps.init()
ps.set_user_callback(callback)
ps.set_build_gui(False)
ps.set_give_focus_on_show(True)
ps.show()

