import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size,
        enc_hidden_size,
        dec_hidden_size,
        padding_idx,
        dropout_ratio,
        device,
        bidirectional=False,
        num_layers=1,
    ):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = enc_hidden_size
        self.device = device
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(
            embedding_size,
            enc_hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=dropout_ratio,
            bidirectional=bidirectional,
        )
        self.encoder2decoder_h = nn.Linear(enc_hidden_size * self.num_directions, dec_hidden_size)
        self.encoder2decoder_c = nn.Linear(enc_hidden_size * self.num_directions, dec_hidden_size)
        self.tanh = nn.Tanh()

    def init_state(self, inputs):
        """Initialize to zero cell states and hidden states

        Args:
            inputs (torch.LongTensor): Vocab indices

        Returns:
            h0 (torch.FloatTensor): Initial hidden state
            c0 (torch.FloatTensor): Initial cell state

        Shape:
            Input:
                inputs: (batch_size, max_instr_length), max_instr_length determined by
                         args.max_input_length
            Output:
                h0: (num_layers * num_directions, batch_size, hidden_size)
                c0: (num_layers * num_directions, batch_size, hidden_size)
        """

        batch_size = inputs.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size,
                         device=self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size,
                         device=self.device)
        return h0, c0

    def forward(self, inputs, lengths):
        """
        Args:
            inputs (torch.LongTensor): Vocab indices
            lengths (torch.LongTensor): Sequence lengths for dynamic batching

        Returns:
            ctx (torch.FloatTensor): Encoder hidden state sequence
            h (torch.FloatTensor): Hidden state initialization for a decoder
            c (torch.FloatTensor): Memory cell initialization for a decoder

        Shape:
            Input:
                inputs: (batch_size, max_instr_length), max_instr_length determined by
                         args.max_input_length
                lengths: (batch_size)
            Output:
                ctx: (batch_size, max_seq_length, enc_hidden_size*num_directions)
                h: (batch_size, dec_hidden_size)
                c: (batch_size, dec_hidden_size)
        """

        # shape: (batch_size, max_instr_length, embedding_size)
        embeds = self.embedding(inputs)

        # shape: h0 (num_layers * num_directions, batch_size, hidden_size)
        # shape: c0 (num_layers * num_directions, batch_size, hidden_size)
        h0, c0 = self.init_state(inputs)

        # if RuntimeError, it's a bug: # https://github.com/pytorch/pytorch/issues/16542
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)

        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        # shape: h_t (batch_size, num_directions * enc_hidden_size)
        # shape: c_t (batch_size, num_directions * enc_hidden_size)
        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]  # (batch, hidden_size)

        # shape: h (batch_size, dec_hidden_size)
        # shape: c (batch_size, dec_hidden_size)
        h = self.tanh(self.encoder2decoder_h(h_t))
        c = self.encoder2decoder_c(c_t)

        # shape: ctx (batch_size, max_seq_length, enc_hidden_size*num_directions)
        # shape: lengths (batch_size)
        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        return ctx, h, c


class InstructionEncoder(EncoderLSTM):
    """Encodes navigation instructions, returning hidden state context (for
       attention methods) and a decoder initial state."""

    def __init__(self, args):
        super(InstructionEncoder, self).__init__(
            args.vocab_size,
            args.word_embedding_size,
            args.enc_hidden_size,
            args.dec_hidden_size,
            args.enc_padding_idx,
            args.enc_dropout_ratio,
            args.device,
            args.bidirectional,
            args.enc_lstm_layers,
        )
