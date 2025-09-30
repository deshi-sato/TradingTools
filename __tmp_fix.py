# -*- coding: utf-8 -*-
from pathlib import Path
path = Path('scripts/stream_microbatch.py')
text = path.read_text(encoding='utf-8')
start = text.find('"""')
if start == -1:
    raise SystemExit('docstring start not found')
end = text.find('"""', start + 3)
if end == -1:
    raise SystemExit('docstring end not found')
doc_body = "\n".join([
    'stream_microbatch.py',
    '',
    'WebSocket CurrentPrice を取り込み、tick_batch / features_stream を小バッチ更新するスクリプト。',
    '',
    '- CurrentPrice (PUSH via kabu WS) -> tick_batch',
    '- features_stream は既存 orderbook_snapshot の直近レコードを参照して生成（本スクリプトは orderbook_snapshot を更新しない）',
    '依存:',
    '  pip install websocket-client',
    '設定:',
    '  -Config JSON (symbols, host, port, token, market_window, window_ms, db_path, price_guard, tick_queue_max)',
])
new_doc = '"""' + doc_body + '\n"""'
text = text[:start] + new_doc + text[end+3:]
text = text.replace('        # 紁E��数量�E推定！EradingVolume 差刁E��E        tv = obj.get("TradingVolume") or obj.get("Volume")\n', '        # TradingVolume の差分から約定数量を推定\n        tv = obj.get("TradingVolume") or obj.get("Volume")\n')
text = text.replace('    # 板は DBに既に蓁E��されてぁE��前提で、features_stream 生�E時に「直近�E板」を参�E\n', '    # features_stream は蓄積済みの板スナップショットを前提に最新値を参照\n')
text = text.replace('                    # チE��チE��雁E��E                    upt=dwn=0; vol=0.0\n', '                    # ウィンドウ内のティックを走査してアップ/ダウン件数と出来高を集計\n                    upt=dwn=0; vol=0.0\n')
text = text.replace('                    # 直近�E板�E�EB参�Eのみ�E�E                    ob = latest_board_from_db(s, ts_end_iso)\n', '                    # 直近の板スナップショットを引き当てて指標を補完\n                    ob = latest_board_from_db(s, ts_end_iso)\n')
path.write_text(text, encoding='utf-8')
