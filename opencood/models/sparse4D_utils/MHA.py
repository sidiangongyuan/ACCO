import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self,cfg):
        super(MultiheadAttention, self).__init__()

        self.embed_dims = cfg['embed_dims']
        self.num_heads = cfg['num_heads']
        self.batch_first = cfg['batch_first']

        self.attn_drop = cfg['dropout']

        self.attn = nn.MultiheadAttention(self.embed_dims, self.num_heads, self.attn_drop,self.batch_first)

        self.proj_drop = nn.Dropout(self.attn_drop)
        self.dropout_layer = nn.Dropout(self.attn_drop)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):

        if key is None:
            # self-attention
            key = query

        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos


        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)


        if key_padding_mask is not None:
            ignore_batch = key_padding_mask.all(dim=1)
            out = torch.zeros_like(identity)
            if not ignore_batch.all():
                valid_query = query.transpose(0, 1)[~ignore_batch].transpose(0, 1)
                valid_key = key.transpose(0, 1)[~ignore_batch].transpose(0, 1)
                valid_value = value.transpose(0, 1)[~ignore_batch].transpose(0, 1)
                valid_key_padding_mask = None if key_padding_mask is None else key_padding_mask[~ignore_batch]

                expanded_ignore_batch = ignore_batch.unsqueeze(1).repeat(1, self.num_heads).reshape(-1)
                valid_attn_mask_indices = ~expanded_ignore_batch
                valid_attn_mask = attn_mask[valid_attn_mask_indices]
                valid_out = self.attn(
                    query=valid_query,
                    key=valid_key,
                    value=valid_value,
                    attn_mask=valid_attn_mask,
                    key_padding_mask=valid_key_padding_mask)[0]

                out[~ignore_batch] = valid_out.transpose(0, 1)
            return identity + self.dropout_layer(self.proj_drop(out))
        else:

            out = self.attn(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

