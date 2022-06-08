import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture


@register_model('lstm')
class LSTMModel(Seq2SeqModel):
    """ Defines the sequence-to-sequence model class. """

    def __init__(self,
                 encoder,
                 decoder):

        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-embed-dim', type=int, help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-hidden-size', type=int, help='encoder hidden size')
        parser.add_argument('--encoder-num-layers', type=int, help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', help='bidirectional encoder')
        parser.add_argument('--encoder-dropout-in', help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', help='dropout probability for encoder output')

        parser.add_argument('--decoder-embed-dim', type=int, help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, help='decoder hidden size')
        parser.add_argument('--decoder-num-layers', type=int, help='number of decoder layers')
        parser.add_argument('--decoder-dropout-in', type=float, help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, help='dropout probability for decoder output')
        parser.add_argument('--decoder-use-attention', help='decoder attention')
        parser.add_argument('--decoder-use-lexical-model', help='toggle for the lexical model')

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """ Constructs the model. """
        base_architecture(args)
        encoder_pretrained_embedding = None
        decoder_pretrained_embedding = None

        # Load pre-trained embeddings, if desired
        if args.encoder_embed_path:
            encoder_pretrained_embedding = utils.load_embedding(args.encoder_embed_path, src_dict)
        if args.decoder_embed_path:
            decoder_pretrained_embedding = utils.load_embedding(args.decoder_embed_path, tgt_dict)

        # Construct the encoder
        encoder = LSTMEncoder(dictionary=src_dict,
                              embed_dim=args.encoder_embed_dim,
                              hidden_size=args.encoder_hidden_size,
                              num_layers=args.encoder_num_layers,
                              bidirectional=bool(eval(args.encoder_bidirectional)),
                              dropout_in=args.encoder_dropout_in,
                              dropout_out=args.encoder_dropout_out,
                              pretrained_embedding=encoder_pretrained_embedding)

        # Construct the decoder
        decoder = LSTMDecoder(dictionary=tgt_dict,
                              embed_dim=args.decoder_embed_dim,
                              hidden_size=args.decoder_hidden_size,
                              num_layers=args.decoder_num_layers,
                              dropout_in=args.decoder_dropout_in,
                              dropout_out=args.decoder_dropout_out,
                              pretrained_embedding=decoder_pretrained_embedding,
                              use_attention=bool(eval(args.decoder_use_attention)),
                              use_lexical_model=bool(eval(args.decoder_use_lexical_model)))
        return cls(encoder, decoder)


class LSTMEncoder(Seq2SeqEncoder):
    """ Defines the encoder class. """

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=64,
                 num_layers=1,
                 bidirectional=True,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None):

        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_dim = 2 * hidden_size if bidirectional else hidden_size

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        dropout_lstm = dropout_out if num_layers > 1 else 0.
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_lstm,
                            bidirectional=bool(bidirectional))

    def forward(self, src_tokens, src_lengths):
        """ Performs a single forward pass through the instantiated encoder sub-network. """
        # Embed tokens and apply dropout
        batch_size, src_time_steps = src_tokens.size()
        src_embeddings = self.embedding(src_tokens)
        _src_embeddings = F.dropout(src_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [src_time_steps, batch_size, num_features]
        src_embeddings = _src_embeddings.transpose(0, 1)

        # Pack embedded tokens into a PackedSequence
        packed_source_embeddings = nn.utils.rnn.pack_padded_sequence(src_embeddings, src_lengths.data.tolist())

        # Pass source input through the recurrent layer(s)
        if self.bidirectional:
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
        else:
            state_size = self.num_layers, batch_size, self.hidden_size

        hidden_initial = src_embeddings.new_zeros(*state_size)
        context_initial = src_embeddings.new_zeros(*state_size)

        packed_outputs, (final_hidden_states, final_cell_states) = self.lstm(packed_source_embeddings,
                                                                             (hidden_initial, context_initial))

        # Unpack LSTM outputs and optionally apply dropout (dropout currently disabled)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=0.)
        lstm_output = F.dropout(lstm_output, p=self.dropout_out, training=self.training)
        assert list(lstm_output.size()) == [src_time_steps, batch_size, self.output_dim]  # sanity check

        '''
        ___QUESTION-1-DESCRIBE-A-START___
        1.  Add tensor shape annotation to each of the output tensor
            output tensor means tensors on the left hand side of "="
            e.g., 
                sent_tensor = create_sentence_tensor(...) 
                # sent_tensor.size = [batch, sent_len, hidden]
        2.  Describe what happens when self.bidirectional is set to True. 
        3.  What is the difference between final_hidden_states and final_cell_states?
        '''
        if self.bidirectional:
            def combine_directions(outs):
                return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim=2)
            final_hidden_states = combine_directions(final_hidden_states)
            #print('final hidden states: {}'.format(final_hidden_states.shape))
            final_cell_states = combine_directions(final_cell_states)            
            #print('final cell states: {}'.format(final_cell_states.shape))
            
            #print("bidirectional: {}".format(self.bidirectional))
            #
            """
            1.
            final_hidden_states.size = [num_layers, batch_size, output_size] 
            BIDIRECTIONAL CASE: final_hidden_states.size = [num_layers, batch_size, 2*output_size]
            
            2.
            final_cell_states.size = [num_layers, batch_size, output_size] 
            BIDIRECTIONAL CASE: final_cell_states.size = [num_layers, batch_size, 2*output_size]
            
            3.
            final_cell_states and final_hidden_states have the same dimesions 
            and the difference between them is that while the final hidden state flows
            to the next hidden layer or to the output, the final hidden cell flows
            into the next timestep representing the memory of the LSTM.
            """
            
        '''___QUESTION-1-DESCRIBE-A-END___'''

        # Generate mask zeroing-out padded positions in encoder inputs
        src_mask = src_tokens.eq(self.dictionary.pad_idx)

        return {'src_embeddings': _src_embeddings.transpose(0, 1),
                'src_out': (lstm_output, final_hidden_states, final_cell_states),
                'src_mask': src_mask if src_mask.any() else None}


class AttentionLayer(nn.Module):
    """ Defines the attention layer class. Uses Luong's global attention with the general scoring function. """
    def __init__(self, input_dims, output_dims):
        super().__init__()
        # Scoring method is 'general'
        self.src_projection = nn.Linear(input_dims, output_dims, bias=False)
        self.context_plus_hidden_projection = nn.Linear(input_dims + output_dims, output_dims, bias=False)

    def forward(self, tgt_input, encoder_out, src_mask):
        # tgt_input has shape = [batch_size, input_dims]
        # encoder_out has shape = [src_time_steps, batch_size, output_dims]
        # src_mask has shape = [batch_size, src_time_steps]

        # Get attention scores
        # [batch_size, src_time_steps, output_dims]
        encoder_out = encoder_out.transpose(1, 0)

        # [batch_size, 1, src_time_steps]
        attn_scores = self.score(tgt_input, encoder_out)

        '''
        ___QUESTION-1-DESCRIBE-B-START___
        1.  Add tensor shape annotation to each of the output tensor
            output tensor means tensors on the left hand side of "="
            e.g., 
                sent_tensor = create_sentence_tensor(...) 
                # sent_tensor.size = [batch, sent_len, hidden]
        2.  Describe how the attention context vector is calculated. 
        3.  Why do we need to apply a mask to the attention scores?
        '''
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(dim=1)
            attn_scores.masked_fill_(src_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        #print('Attn weights: {}'.format(attn_weights.shape))
        attn_context = torch.bmm(attn_weights, encoder_out).squeeze(dim=1)

        context_plus_hidden = torch.cat([tgt_input, attn_context], dim=1)
        attn_out = torch.tanh(self.context_plus_hidden_projection(context_plus_hidden))
        #print('Attn out: {}'.format(attn_out.shape))
        #print(attn_weights.squeeze(dim=1).shape)
        #print("bidirectional: {}".format(self.bidirectional))
        
        """
        1.
        attn_weights: [batch_size, seq_len] --> AFTER doing squeeze, i.e. removing dim-1 from the tensor
        attn_out: [batch_size, output_size]
        BIDIRECTIONAL CASE: attn_out: [batch_size, 2*output_size]
        
        2.
        The attention context vector is the result of the dot product between the attention weights and
        the encoder's output, i.e. the weighted average of the input sequence
        
        3.
        We need to apply a mask to prevent the attention from cheating, i.e. we only attend to words in the 
        sequence that are before "t", the tokens from "t+1" and forward are masked. 
        
        """

        '''___QUESTION-1-DESCRIBE-B-END___'''

        return attn_out, attn_weights.squeeze(dim=1)

    def score(self, tgt_input, encoder_out):
        """ Computes attention scores. """

        '''
        ___QUESTION-1-DESCRIBE-C-START___
        1.  Add tensor shape annotation to each of the output tensor
        2.  How are attention scores calculated? 
        3.  What role does batch matrix multiplication (i.e. torch.bmm()) play in aligning encoder and decoder representations?
        '''
        projected_encoder_out = self.src_projection(encoder_out).transpose(2, 1)
        attn_scores = torch.bmm(tgt_input.unsqueeze(dim=1), projected_encoder_out)
        
        
        #print("tgt_input shape: {}".format(tgt_input.shape))
        #print("projected encoder out: {}".format(projected_encoder_out.shape))
        #print("attn scores: {}".format(attn_scores.shape))
        
        """
        1.
        projected_encoder_out: [batch_size, hidden_dim, seq_len]
        attn_scores: [batch_size, 1, seq_len]
        
        
        2.
        The attention scores are the result of a matrix multiplication between the input sequence and the
        encoder's output (transposed for the dimensions to properly match)
        
        3.
        Since we are doing tensor multiplication, torch.bmm() helps us to align the encoder and encoder representations
        by helping us to get a resultant tensor that IS NOT  shaped by the hidden layer's dimension (e.g. 64 or 128).
        This batch matrix multiplication allows us to get a tensor whose first dimension is the batch size and the 
        final dimension is the sequence length. 
        
        """
        
        
        '''___QUESTION-1-DESCRIBE-C-END___'''

        return attn_scores


class LSTMDecoder(Seq2SeqDecoder):
    """ Defines the decoder class. """

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=128,
                 num_layers=1,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None,
                 use_attention=True,
                 use_lexical_model=False):

        super().__init__(dictionary)

        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        # Define decoder layers and modules
        self.attention = AttentionLayer(hidden_size, hidden_size) if use_attention else None

        self.layers = nn.ModuleList([nn.LSTMCell(
            input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
            hidden_size=hidden_size)
            for layer in range(num_layers)])

        self.final_projection = nn.Linear(hidden_size, len(dictionary))

        self.use_lexical_model = use_lexical_model
        if self.use_lexical_model:
            # __QUESTION-5: Add parts of decoder architecture corresponding to the LEXICAL MODEL here
            # TODO: --------------------------------------------------------------------- CUT
            self.ff_lex = nn.Linear(embed_dim, embed_dim, bias=False)
            self.tanh = nn.Tanh()
            #self.hh_lex = nn.Linear(hidden_size, len(dictionary))
            
            self.decoder_lex = nn.Linear(embed_dim, len(dictionary), bias=True)
            
            pass
            # TODO: --------------------------------------------------------------------- /CUT

    def forward(self, tgt_inputs, encoder_out, incremental_state=None):
        """ Performs the forward pass through the instantiated model. """
        # Optionally, feed decoder input token-by-token
        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]

        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Recover encoder input
        src_embeddings = encoder_out['src_embeddings']

        src_out, src_hidden_states, src_cell_states = encoder_out['src_out']
        src_mask = encoder_out['src_mask']
        src_time_steps = src_out.size(0)

        # Embed target tokens and apply dropout
        batch_size, tgt_time_steps = tgt_inputs.size()
        tgt_embeddings = self.embedding(tgt_inputs)
        tgt_embeddings = F.dropout(tgt_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, tgt_time_steps, num_features] -> [tgt_time_steps, batch_size, num_features]
        tgt_embeddings = tgt_embeddings.transpose(0, 1)

        # Initialize previous states (or retrieve from cache during incremental generation)
        '''
        ___QUESTION-1-DESCRIBE-D-START___
        1.  Add tensor shape annotation to each of the output tensor
        2.  Describe how the decoder state is initialized. 
        3.  When is cached_state == None? 
        4.  What role does input_feed play?
        '''
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        
        #print(cached_state.shape)
        
        if cached_state is not None:
            tgt_hidden_states, tgt_cell_states, input_feed = cached_state
            
            #print("tgt_hidden states shape: {}".format(tgt_hidden_states.shape))
            #print("tgt_cell states shape: {}".format(tgt_cell_states.shape))
            #print("input feed shape: {}".format(input_feed.shape))
        else:
            tgt_hidden_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            tgt_cell_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            input_feed = tgt_embeddings.data.new(batch_size, self.hidden_size).zero_()
            
            #print("tgt_hidden states shape: {}".format(tgt_hidden_states[0].shape))
            #print("tgt_cell states shape: {}".format(tgt_cell_states[0].shape))
            #print("input feed shape: {}".format(input_feed.shape))
            
        """
        1.
        tgt_hidden_states: [batch_size, hidden_dim]
        tgt_cell_states: [batch_size, hidden_dim]
        input_feed: [batch_size, hidden_dim]
        
        2. 
        The cell state and hidden state tensors are simply initialized to zeros.
        
        3.
        cached_state == None at inference time, i.e. when we predict the next word, we
        must recover the previous cell states and hidden states. During training, if they don't exist,
        the tensors are simply initialized to zeros, as described above.
        
        4.
        The input feed is the the tensor that contains the embeddings generated by the Encoder. i.e. the input
        to the Decoder which is necessary to predict the next token in the sequence.
        
        
        
        """
            
        '''___QUESTION-1-DESCRIBE-D-END___'''

        # Initialize attention output node
        attn_weights = tgt_embeddings.data.new(batch_size, tgt_time_steps, src_time_steps).zero_()
        rnn_outputs = []

        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Cache lexical context vectors per translation time-step
        lexical_contexts = []

        for j in range(tgt_time_steps):
            # Concatenate the current token embedding with output from previous time step (i.e. 'input feeding')
            lstm_input = torch.cat([tgt_embeddings[j, :, :], input_feed], dim=1)

            for layer_id, rnn_layer in enumerate(self.layers):
                # Pass target input through the recurrent layer(s)
                tgt_hidden_states[layer_id], tgt_cell_states[layer_id] = \
                    rnn_layer(lstm_input, (tgt_hidden_states[layer_id], tgt_cell_states[layer_id]))

                # Current hidden state becomes input to the subsequent layer; apply dropout
                lstm_input = F.dropout(tgt_hidden_states[layer_id], p=self.dropout_out, training=self.training)

            '''
            ___QUESTION-1-DESCRIBE-E-START___
            1.  Add tensor shape annotation to each of the output tensor
            2.  How is attention integrated into the decoder? 
            3.  Why is the attention function given the previous target state as one of its inputs? 
            4.  What is the purpose of the dropout layer?
            '''
            #print("source embeddings shape: {}".format(src_embeddings.shape))
            #print("source embeddings shape transposed: {}".format(torch.transpose(src_embeddings,0,1).shape))
            #print("attention shape: {}".format(attn_weights.shape))


            
            if self.attention is None:
                input_feed = tgt_hidden_states[-1]
            else:
                input_feed, step_attn_weights = self.attention(tgt_hidden_states[-1], src_out, src_mask)
                attn_weights[:, j, :] = step_attn_weights
                
                #print("step attention shape: {}".format(step_attn_weights.shape))
                #print("source embeddings shape: {}".format(src_embeddings.shape))
                

                if self.use_lexical_model:
                    # __QUESTION-5: Compute and collect LEXICAL MODEL context vectors here
                    # TODO: --------------------------------------------------------------------- CUT
                    # SOURCE embeddings: > [src_time_steps, batch_size, num_features]
                    # att weights: [batch_size, seq_len, -]
                    weighted_sum = torch.bmm(step_attn_weights.unsqueeze(dim=1), 
                                             torch.transpose(src_embeddings,0,1))
                    
                    #print("weighted sum: {}".format(weighted_sum.shape))
                    
                    # Apply hyperbolic tangent:
                    
                    weighted_sum_tanh = self.tanh(weighted_sum)
                    
                    
                    # Append to list:
                    lexical_contexts.append(weighted_sum_tanh)
                    
                    """
                    #weighted sum of source embeddings, with att weights
                    weighted_sum = [] #empty vector of length vocab size ????
                    for k in range(seq_len): # que hacer con la dimension del bach size??
                        weigted_sum += att_weights[k,j] * src_embeddings[k] #check dimension ??????????
                    """
                    #pass
                    # TODO: --------------------------------------------------------------------- /CUT

            input_feed = F.dropout(input_feed, p=self.dropout_out, training=self.training)
            rnn_outputs.append(input_feed)
            
            
            #print("input_feed shape: {}".format(input_feed.shape))
            #print("rnn_outputs shape: {}".format(len(rnn_outputs)))
            #print("attention_weights shape: {}".format(attn_weights.shape))
            
            """
            1.
            input_feed: [batch_size, 2*hidden_size]
            rnn_outputs: not a tensor but a list of len() = sequence length
            attn_weights: [batch_size, tgt_size, src_size]
            
            2.
            For every timestep, a new attention "vector" is calculated and added to the attn_weights tensor, 
            along the dim-1 axis. This is necessary because at every time step  the last layer's state will 
            change and the masking will also be adjusted according to the current time step. That's why the
            attn weights are calculated at every step of this for-loop.
            
            3.
            The attention function is given the previous target state in order to "attend"
            to it as part of the context vector.

            4.
            The purpose of the dropout layer is to randomly replace some of the attention values with zeros. 
            This introduction of stochastic noise should help the model to generalize better, since
            it will be less likely to overfit, as it will not "memorize" the tokens that frequently align 
            with either low or high attention scores.            
            
            
            """
            
            
            
            '''___QUESTION-1-DESCRIBE-E-END___'''

        # Cache previous states (only used during incremental, auto-regressive generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (tgt_hidden_states, tgt_cell_states, input_feed))

        # Collect outputs across time steps
        decoder_output = torch.cat(rnn_outputs, dim=0).view(tgt_time_steps, batch_size, self.hidden_size)

        # Transpose batch back: [tgt_time_steps, batch_size, num_features] -> [batch_size, tgt_time_steps, num_features]
        decoder_output = decoder_output.transpose(0, 1)

        # Final projection
        decoder_output = self.final_projection(decoder_output)

        if self.use_lexical_model:
            # __QUESTION-5: Incorporate the LEXICAL MODEL into the prediction of target tokens here
            # TODO: --------------------------------------------------------------------- CUT
            
            lex_tensor = torch.cat(lexical_contexts, 1)
            
            print("lex_tensor shape: {}".format(lex_tensor.shape))
            print("decoder_output shape: {}".format(decoder_output.shape))
            
            
            # h_lex: FFNN with skip connections
            h_lex = self.tanh(self.ff_lex(lex_tensor)) + lex_tensor
            
            
            # Redefine the decoder ouput to also include h_lex
            decoder_output = decoder_output + self.decoder_lex(h_lex)
            
            #pass
            # TODO: --------------------------------------------------------------------- /CUT

        return decoder_output, attn_weights


@register_model_architecture('lstm', 'lstm')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 64)
    args.encoder_num_layers = getattr(args, 'encoder_num_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', 'True')
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.25)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.25)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 128)
    args.decoder_num_layers = getattr(args, 'decoder_num_layers', 1)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.25)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.25)
    args.decoder_use_attention = getattr(args, 'decoder_use_attention', 'True')
    args.decoder_use_lexical_model = getattr(args, 'decoder_use_lexical_model', 'False')
