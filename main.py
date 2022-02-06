from model import *
from data import *
import utils.data_tools as dt
from torch import optim





class_graph_data = classGraphDataset(root=DATA_HOME) # 40 samples
summary = torch.load(DATA_HOME + config['data']['field_summary'])
summary_token = dt.load(DATA_HOME + config['data']['raw_base_summary'])
summary_idx = summary.process([summary_token[0]]).T
print(summary_idx)
print(summary_idx[0].shape)
# data_loader = DataLoader(class_graph_data, batch_size=20, shuffle=True)
# for batch in data_loader:
#     print(batch)
#     # print(batch)
#     # print(batch.x)
#     break
globalencoder = GlobalEncoder(
    vocab_size=1704+1, embed_size=128, GRU_num_inputs=128, GRU_num_hiddens=128, GRU_num_layers=1,
    GAT_num_layers=4, GAT_in_features=256, GAT_out_features=256, GAT_dropout=0.1)
glo_enc_outputs, loc_enc_hiddens, loc_enc_outputs = globalencoder(class_graph_data[0])


decoder = Decoder(vocab_size=1704+1, embed_size=128, num_inputs=128, num_hiddens=128)
X = torch.zeros((1, 20), dtype=torch.long)
pre = decoder(summary_idx[0].unsqueeze(0), glo_enc_outputs, loc_enc_hiddens, loc_enc_outputs)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Y = torch.zeros((40, 4, 16), dtype=torch.long)
# con = Y[-1].repeat(X.shape[0],1,1)
# print(Y.shape, con.shape)
# print(torch.cat((Y, con), 2).shape)
