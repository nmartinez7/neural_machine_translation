#   Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017) <https://arxiv.org/abs/1706.03762>`.
#   This is a basic version intended for demonstration ONLY as part of INFR11157 course (NLU+).
#   Tom Sherborne, University of Edinburgh, January 2020
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture
from seq2seq.models.transformer_helper import TransformerEncoderLayer, TransformerDecoderLayer, PositionalEmbedding, generate_embedding, fill_with_neg_inf

DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512


@register_model('transformer')
class TransformerModel(Seq2SeqModel):
    """
    Transformer Model Class. Inherits from Seq2SeqModel and calls TransformerEncoder and TransformerDecoder submodels.
    """
    def __init__(self,
                 args,
                 encoder,
                 decoder):
        super().__init__(encoder, decoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D', help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N', help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N', help='num decoder attention heads')
        parser.add_argument('--no-scale-embedding', action='store_true', help='if True, dont scale embeddings')

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """Construct model. """
        base_architecture(args)

        if getattr(args, 'max_source_positions', None) is None:
            args.max_src_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_tgt_positions = DEFAULT_MAX_TARGET_POSITIONS

        # Transformer Encoder
        encoder = TransformerEncoder(args, src_dict)
        decoder = TransformerDecoder(args, tgt_dict)
        return cls(args, encoder, decoder)


class TransformerEncoder(Seq2SeqEncoder):
    """ Defines an encoder class. """

    def __init__(self,
                 args,
                 dictionary):

        super().__init__(dictionary)

        self.dropout = args.dropout
        self.embed_dim = args.encoder_embed_dim
        self.padding_idx = dictionary.pad_idx
        self.max_src_positions = args.max_src_positions
        self.embedding = generate_embedding(len(dictionary), self.embed_dim, dictionary.pad_idx)
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.embed_dim)

        self.embed_positions = PositionalEmbedding(
            self.embed_dim, padding_idx=self.padding_idx, init_size=self.max_src_positions + self.padding_idx + 1
        )

        self.layers = nn.ModuleList([])

        # Generate N identical Encoder Layers
        self.layers.extend([
            TransformerEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])

    def forward(self, src_tokens, src_lengths):
        # Embed tokens indices
        embeddings = self.embed_scale * self.embedding(src_tokens)

        # Clone for output state
        src_embeddings = embeddings.clone()

        '''
        ___QUESTION-6-DESCRIBE-A-START___
        1.  Add tensor shape annotation to each of the output tensor
        2.  What is the purpose of the positional embeddings in the encoder and decoder? 
        3.  Why can't we use only the embeddings similar to for the LSTM? 
        '''
        embeddings += self.embed_positions(src_tokens)
        
        #print("embeddings size: {}".format(embeddings.shape))
        #print("scr_tokens size: {}".format(src_tokens.shape))
        
        """
        1. 
        embeddings: [batch_size, src_lengths, embed_dim]
        
        2.
        By definition, the Transformer is permutation invariant, i.e. if we shuffle up words, the translation would be the same.
        However, we know that word order does matter in linguistics, thus we use positional embeddings
        for the model to be somewhat sensitive to word order.        
        
        3.
        The LSTM processes the input follwing its natural order. Therefore, LSTM
        is naturally sensitive to word order. However, this does not hold in the 
        transformer model, so positional embeddings are required, to feed the model
        with this information.
        
        """
        
        
        '''
        ___QUESTION-6-DESCRIBE-A-END___
        '''
        forward_state = F.dropout(embeddings, p=self.dropout, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [src_time_steps, batch_size, num_features]
        forward_state = forward_state.transpose(0, 1)

        # Compute padding mask for attention
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # Forward pass through each Transformer Encoder Layer
        for layer in self.layers:
            forward_state = layer(state=forward_state, encoder_padding_mask=encoder_padding_mask)

        return {
            "src_out": forward_state,   # [src_time_steps, batch_size, num_features]
            "src_embeddings": src_embeddings,   # [batch_size, src_time_steps, num_features]
            "src_padding_mask": encoder_padding_mask,   # [batch_size, src_time_steps]
            "src_states": []    # List[]
        }


class TransformerDecoder(Seq2SeqDecoder):
    """ Defines an decoder class. """
    def __init__(self,
                 args,
                 dictionary):

        super().__init__(dictionary)
        self.dropout = args.dropout
        self.embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_embed_dim
        self.padding_idx = dictionary.pad_idx
        self.max_tgt_positions = args.max_tgt_positions

        self.embedding = generate_embedding(len(dictionary), self.embed_dim, dictionary.pad_idx)
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.embed_dim)

        self.embed_positions = PositionalEmbedding(
            self.embed_dim, padding_idx=self.padding_idx, init_size=self.max_tgt_positions + self.padding_idx + 1
        )

        self.layers = nn.ModuleList([])

        # Generate N identical Decoder Layers
        self.layers.extend([
            TransformerDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])

        self.embed_out = nn.Linear(self.output_embed_dim, len(dictionary))

        nn.init.normal_(self.embed_out.weight, mean=0, std=self.output_embed_dim ** -0.5)

    def forward(self, tgt_inputs, encoder_out=None, incremental_state=None, features_only=False):
        # Embed positions
        positions = self.embed_positions(tgt_inputs, incremental_state=incremental_state)

        # Incremental decoding only needs the single previous token
        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        forward_state = self.embed_scale * self.embedding(tgt_inputs)
        forward_state += positions
        forward_state = F.dropout(forward_state, p=self.dropout, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [tgt_time_steps, batch_size, num_features]
        forward_state = forward_state.transpose(0, 1)

        # Generate padding mask
        self_attn_padding_mask = tgt_inputs.eq(self.padding_idx) if tgt_inputs.eq(self.padding_idx).any() else None

        # Forward pass through each Transformer Decode Layer
        attn_state = None
        inner_states = [forward_state]

        for layer_idx, layer in enumerate(self.layers):
            is_attention_layer = layer_idx == len(self.layers) - 1
            encoder_state = encoder_out['src_out'] if encoder_out is not None else None
            '''
            ___QUESTION-6-DESCRIBE-B-START___
            1.  Add tensor shape annotation to each of the output tensor
            2.  What is the purpose of self_attn_mask? 
            3.  Why do we need it in the decoder but not in the encoder?
            4.  Why do we not need a mask for incremental decoding?
            '''
            self_attn_mask = self.buffered_future_mask(forward_state) if incremental_state is None else None
            
            
            #print("self_attn_mask: {}".format(self_attn_mask.shape))
            
            """
            1. Add tensor shape annotation to each of the output tensor
            self_attn_mask: [src_lengths, src_lengths]
            
            2. What is the purpose of self_attn_mask? 
            Transformers use self-attention to understand how previous words are related to
            the current word in a sentence. 
            Masking the input keeps the model from "cheating",
            i.e. attending to future words would be cheating. After masking
            is applied, we guarantee that the model can only attend to the previous words in the sequence. 
            
            3. Why do we need it in the decoder but not in the encoder?
            Because in the encoder we are mapping the entire word sequence to a latent vector.
            It is this latent vector that will be passed on as the decoder's input. We want this vector
            to be a rich representation of the sentence's meaning. Masking is not necessary since the
            vector has to represent the whole sentence, and it would actually harm model performance.
            
            4. Why do we not need a mask for incremental decoding?
            Because we have access to the previous generated tokens. Therefore, there is no possibility
            for the self-attention to cheat since there are no future tokens to attend to.
            
            """       
            
            
            
            '''
            ___QUESTION-6-DESCRIBE-B-END___
            '''

            forward_state, layer_attn = layer(state=forward_state,
                                              encoder_out=encoder_state,
                                              self_attn_mask=self_attn_mask,
                                              self_attn_padding_mask=self_attn_padding_mask,
                                              need_attn=is_attention_layer,
                                              need_head_weights=is_attention_layer)
            inner_states.append(forward_state)

            if layer_attn is not None and is_attention_layer:
                attn_state = layer_attn.float()

        if attn_state is not None:
            attn_state = attn_state.mean(dim=0)

        forward_state = forward_state.transpose(0, 1)

        # Project into output layer
        if not features_only:
            '''
            ___QUESTION-6-DESCRIBE-C-START___
            1.  Add tensor shape annotation to each of the output tensor
            2.  Why do we need a linear projection after the decoder layers? 
            3.  What is the dimensionality of forward_state after this line? 
            4.  What would the output represent if features_only=True?
            '''
            #print("forward_state before: {}".format(forward_state.shape))
            forward_state = self.embed_out(forward_state)
            
            #print("forward_state after: {}".format(forward_state.shape))
            #print("tgt inputs: {}".format(tgt_inputs.shape))
            
            """
            1. Add tensor shape annotation to each of the output tensor
            The forward_state before the linear projection has the following dimensions:
            forward_state: [batch_size, tgt_inputs, embed_dim]
            
            2. Why do we need a linear projection after the decoder layers? 
            After the decoder layers, we need a linear projection to transform
            the output from embeddings to target language. The final output has to 
            be words in the target language.
            
            3.  What is the dimensionality of forward_state after this line? 
            After the linear projection the dimensions of the output are the following:
            forward_state: [batch_size, tgt_inputs, vocab_size]
            where vocab_size is the size of the target language vocabulary, English i
            in this case.
            
            4.  What would the output represent if features_only=True?
            If features_only is True, the output will not be projected into the
            target language space, and will be returned in the embedding representation.
            """
            
            
            
            '''
            ___QUESTION-6-DESCRIBE-C-END___
            '''
        return forward_state, {
            "attn_state": attn_state,
            "inner_states": inner_states
        }

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (not hasattr(self, '_future_mask')) or self._future_mask is None or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    # Enc-Dec params [mostly tied for simplicity]
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', args.encoder_layers)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', args.encoder_attention_heads)

    # Dropout and activation
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.2)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)