import torch
from transformers import BertForSequenceClassification, MobileBertForSequenceClassification
from tvm import relay
import tvm


def export_bert_base():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=False)

    batch_size = 1
    seq_len = 128

    input_shapes = [("input_ids", ((batch_size, seq_len), "int64")),
                    ("attention_mask", ((batch_size, seq_len), "int64")),
                    ("token_type_ids", ((batch_size, seq_len), "int64"))]

    inputs = (torch.ones(batch_size, seq_len, dtype=torch.int64),
              torch.ones(batch_size, seq_len, dtype=torch.int64),
              torch.ones(batch_size, seq_len, dtype=torch.int64))

    with torch.no_grad():
        out = model(*inputs)

    script_module = torch.jit.trace(model, inputs).eval()

    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)
    # mod = relay.transform.FastMath()(mod)
    mod = relay.transform.ToMixedPrecision("float16")(mod)

    with open("bert-base-fp16.json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open("bert-base-fp16.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


def export_mobilebert():
    model = MobileBertForSequenceClassification.from_pretrained("lordtt13/emo-mobilebert", return_dict=False)

    # print(model)
    batch_size = 1
    seq_len = 384
    inputs = (torch.ones(batch_size, seq_len, dtype=torch.int64),
              torch.ones(batch_size, seq_len, dtype=torch.int64),
              torch.ones(batch_size, seq_len, dtype=torch.int64))

    input_shapes = [("input_ids", (inputs[0].shape, "int64")),
                    ("attention_mask", (inputs[1].shape, "int64")),
                    ("token_type_ids", (inputs[2].shape, "int64"))]

    with torch.no_grad():
        out = model(*inputs)

    script_module = torch.jit.trace(model, inputs).eval()

    import time
    t1 = time.time()
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)
    t2 = time.time()

    # print(relay.transform.InferType()(mod))
    # mod = relay.transform.FastMath()(mod)
    mod = relay.transform.ToMixedPrecision("float16")(mod)

    with open("mobilebert-fp16.json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open("mobilebert-fp16.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


export_mobilebert()
export_bert_base()
