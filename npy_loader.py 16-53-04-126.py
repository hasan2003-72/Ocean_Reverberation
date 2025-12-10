import numpy as np
from config import Polygon
import os

os.chdir(r"C:\bachelorarbeit-hasan-yesil\ISM_ALL\finnraum")

# def compute_normal(vertices):
#     v1 = vertices[1] - vertices[0]
#     v2 = vertices[2] - vertices[0]
#     n = np.cross(v1, v2)
#     n = n / np.linalg.norm(n)
#     return n

# def npy_to_polygons(room_array, reflexion=0.9):
#     polygons = []
#     for i, surf in enumerate(room_array):
#         vertices = np.array(surf)
#         normal = compute_normal(vertices)
#         point = vertices[0]

#         polygons.append(
#             Polygon(
#                 name=f"Wand_{i}",
#                 vertices=vertices,
#                 normal=normal,
#                 point=point,
#                 reflection=reflexion
#             )
#         )
#     return polygons

# print(os.getcwd())

room = np.load("Room.npy", allow_pickle=True)
print(room.dtype)
print(type(room))

rx = np.load("RxPosition.npy", allow_pickle=True)
print(room.dtype)
print(type(rx))

tx = np.load("TxPosition.npy", allow_pickle=True)
print(room.dtype)
print(type(tx))