from gr_tradinggame.coding.client import Client
from solution import play

client = Client("local-test", password="")
client.test(play)
