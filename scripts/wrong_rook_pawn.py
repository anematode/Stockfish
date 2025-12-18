import chess
import chess.syzygy

# [white king][black king][white bishop][stm]
# 64 * 64 * 64 * 2

output = bytearray(524288)
written = set()

def write_bit(byte_index, bit_offset, val):
	bit_index = byte_index * 8 + bit_offset
	if bit_index in written:
		print("???")
		exit(1)
	written.add(bit_index)
	output[byte_index] |= (val << bit_offset)



with chess.syzygy.open_tablebase("/home/toystory/Desktop/tb/src") as tablebase:
	for wk in range(64):
		print("wk", wk)
		for bk in range(64):
			for wbishop in range(64):
				for pawn_rank in range(1,7):
					for stm in "wb":
						board = chess.Board.empty()
						board.set_piece_at(chess.Square(wk), chess.Piece(chess.KING, chess.WHITE))
						board.set_piece_at(chess.Square(bk), chess.Piece(chess.KING, chess.BLACK))
						board.set_piece_at(chess.Square(wbishop), chess.Piece(chess.BISHOP, chess.WHITE))
						pawn_file = 7 if (1 << wbishop) & chess.BB_LIGHT_SQUARES else 0
						board.set_piece_at(chess.Square(pawn_file + pawn_rank * 8), chess.Piece(chess.PAWN, chess.WHITE))
						board.turn = stm == "w"
						if not board.is_valid():
							continue
						wdl = tablebase.probe_wdl(board)
						white_wins = wdl != 0
						write_bit(wk * 64 * 64 * 2 + bk * 64 * 2 + wbishop * 2 + (1 if stm == "b" else 0), pawn_rank, white_wins)

with open('wrong_rook_pawn.dat', 'wb') as f:
	f.write(output)

