from pytools import memoize_method
import torch
import torch.nn.functional as F
import pytorch_pretrained_bert
import modeling_util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomBertModel(pytorch_pretrained_bert.BertModel):
    """
    Based on pytorch_pretrained_bert.BertModel, but also outputs un-contextualized embeddings.
    """

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Based on pytorch_pretrained_bert.BertModel
        """
        embedding_output = self.embeddings(input_ids, token_type_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=True)

        return [embedding_output] + encoded_layers

class BertRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.CHANNELS = 12 + 1  # from bert-base-uncased
        self.BERT_SIZE = 768  # from bert-base-uncased
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask, customBert=None):
        BATCH, QLEN = query_tok.shape
        DIFF = 3  # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, doc_toks, SEPS, query_toks, SEPS], dim=1)
        mask = torch.cat([ONES, doc_mask, ONES, query_mask, ONES], dim=1)
        # segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        segment_ids = torch.cat([NILS] * (2 + doc_toks.shape[1]) + [ONES] * (1 + QLEN), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # execute BERT model
        if not customBert:
            result = self.bert(toks, segment_ids.long(), mask)
        else:
            result = customBert(toks, segment_ids.long(), mask)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN + 1] for r in result]
        doc_results = [r[:, QLEN + 2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results



class CrossBert(BertRanker):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        self.cls2 = torch.nn.Linear(self.BERT_SIZE, 1)
        self.clsAll = torch.nn.Linear(2, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, wiki_tok, wiki_mask, question_tok, question_mask):
        cls_query_tok, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        cls_doc_tok, _, _ = self.encode_bert(doc_tok, doc_mask, query_tok, query_mask)
        if self.args.mode % 2 == 0:
            cls_wiki_doc_tok, _, _ = self.encode_bert(wiki_tok, wiki_mask, doc_tok, doc_mask)
            cls_doc_wiki_tok, _, _ = self.encode_bert(doc_tok, doc_mask, wiki_tok, wiki_mask)

        if self.args.mode == 1:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            return self.cls(self.dropout(mul))

        elif self.args.mode == 2:
            mul = torch.mul(cls_query_tok[-1], cls_doc_tok[-1])
            mul_wiki = torch.mul(cls_wiki_doc_tok[-1], cls_doc_wiki_tok[-1])
            cat = self.cls(self.dropout(mul))
            cat_wiki = self.cls2(self.dropout(mul_wiki))
            return self.clsAll(torch.cat([cat, cat_wiki], dim=1))

