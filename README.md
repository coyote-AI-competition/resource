# coyote websocket

### how to test in arena(not websocket)

事前に作った AI で遊びたい場合は、以下のように追加してください！

```
 from client.sample_arena_client import SampleClient as SampleClient

    predefs = [
        [SampleClient(player_name="PreAI1", is_ai=True), "PreAI1"],
        [SampleClient(player_name="PreAI2", is_ai=True), "PreAI2"]
    ]

    arena = Arena(total_matches=5, predefined_clients=predefs)
    arena.run()
```

その後実行

```
/$ python3 arena.py
```

### how to test

- server の立ち上げ

```
/server$ python3 server.py
```

- client の立ち上げ

```
/client$ python3 client.py
```

```
Enter room ID (default: test-room):
Enter server port (default: 5000):
Enter player name (default: player1):
Engine.IO connection established
websocket-client package not installed, only polling transport is available
Namespace / is connected
Connected to server
Emitting event "join_room" [/]

Enter command (start, log, exit): Received event "room_message" [/]
Message from server: {'header': 'join_room', 'msg': 'player1 has joined the room!', 'room_id': 'test-room'}
```

これになったら、Enter を押す

```
Enter command (start, log, exit):
```

これが出てくるので、start と入力し、Enter

```
Enter number of games:
```

で対戦の数を入れる

山札内容
-10---------×1
-5----------×2
0-----------×3
1~5---------×4
10----------×3
15----------×2
20----------×1
×2----------×1 100 場の数の合計を二倍する
max→0-------×1 101 場に出ているカードの中で最大の数を１つだけ０にする
0(黒)-------×1 102 ラウンド終了時山札をシャッフルする
?-----------×1 103「コヨーテ」が宣言されたら、山札からあらたに１枚引いてその数に置き換わります


**このコヨーテは六人用です**


#### AI_player_action の引数説明

##### log -> list

turn の数、誰がやったか、action(宣言した数)

```
[{'turn_count': 0, 'turn_player': 'PreAI1', 'action': 8}, {'turn_count': 1, 'turn_player': 'PreAI2', 'action': 9}, {'turn_count': 2, 'turn_player': 'PreAI1', 'action': 14}, {'turn_count': 3, 'turn_player': 'PreAI2', 'action': 17}, {'turn_count': 4, 'turn_player': 'PreAI1', 'action': 25}]
```

##### others_info -> list

name : player の名前
card_info : None(observer 用なので開発者は気にしなくて大丈夫です)
life : その人の life
is_next : 次のターンの人か
is_prev : 前のターンの人か

```
[{'name': 'PreAI2', 'card_info': None, 'life': 2, 'is_next': True, 'is_prev': False}, {'name': 'AI_4', 'card_info': None, 'life': 1, 'is_next': False, 'is_prev': True}]
```

##### sum -> int

自分から見える他の人の sum

##### actions -> int

-1 : coyote するときに返す
その他の数字 : coyote するときに宣言できる数字

```
[-1, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
```
