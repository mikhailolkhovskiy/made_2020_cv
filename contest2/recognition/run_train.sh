python train.py -d ../plates_all -e 100 -b 128 -wd 0.00005 -lr 0.0001 -lrg 0.5 -o ../logs -rdo 0. -v 0.9 -l ../logs_03/cp-best.pth 


# parser.add_argument('-d', '--data_path', dest='data_path', type=str, default=None ,help='path to the data')
# parser.add_argument('--epochs', '-e', dest='epochs', type=int, help='number of train epochs', default=100)
# parser.add_argument('--batch_size', '-b', dest='batch_size', type=int, help='batch size', default=128) # 1o024
# parser.add_argument('--weight_decay', '-wd', dest='weight_decay', type=float, help='weight_decay', default=5e-4)
# parser.add_argument('--lr', '-lr', dest='lr', type=float, help='lr', default=1e-4)
# parser.add_argument('--lr_step', '-lrs', dest='lr_step', type=int, help='lr step', default=None)
# parser.add_argument('--lr_gamma', '-lrg', dest='lr_gamma', type=float, help='lr gamma factor', default=None)
# parser.add_argument('--input_wh', '-wh', dest='input_wh', type=str, help='model input size', default='320x64')
# parser.add_argument('--rnn_dropout', '-rdo', dest='rnn_dropout', type=float, help='rnn dropout p', default=0.1)
# parser.add_argument('--rnn_num_directions', '-rnd', dest='rnn_num_directions', type=int, help='bi', default=1)
# parser.add_argument('--augs', '-a', dest='augs', type=float, help='degree of geometric augs', default=0)
# parser.add_argument('--load', '-l', dest='load', type=str, help='pretrained weights', default=None)
# parser.add_argument('-v', '--val_split', dest='val_split', default=0.8, help='train/val split')
# parser.add_argument('-o', '--output_dir', dest='output_dir', default='/tmp/logs_rec/',
#                     help='dir to save log and models')
