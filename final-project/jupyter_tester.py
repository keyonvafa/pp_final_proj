from final_utils import train_and_test
from datetime import datetime

def make_savedir(K, skip, q, map_estimate, lr, data='nips', most_skip=False):
    timestamp = datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S")
    savedir = 'info/'
    savedir += timestamp + '_' + data + '_' + '_'.join([str(ks) for ks in K])
    savedir += '_' + q
    savedir += '_lr_' + str(lr)
    if skip:
        savedir = savedir + '_skip'
    if map_estimate:
        savedir = savedir + '_map'
    if most_skip:
        savedir = savedir + '_most_skip'
    return savedir

## Section 1A: Map v Random
## Section 1A: MAP
'''K = [100, 50, 10]
skip = False
map_estimate = True
q = 'lognormal'
lr = 0.01
savedir_ln_three_layers_map = make_savedir(K, skip, q, map_estimate, lr)
print("Savedir for 0.01 LN Map estimation on [100, 50, 10]: ", savedir_ln_three_layers_map)
perps, losses, zs, ws = train_and_test(K, savedir_ln_three_layers_map, skip, q=q, lr=lr, n_iter_per_epoch=25000, 
	map_estimate=map_estimate, batch_size=-1, n_test_epoch=50)
print("Savedir for 0.01 LN Map estimation on [100, 50, 10]: ", savedir_ln_three_layers_map)'''
#  info/20171128_230116_nips_100_50_10_lognormal_lr_0.01_map

## Section 1B: Random
'''K = [100, 50, 10]
skip = False
map_estimate = False
q = 'lognormal'
lr = 0.01
savedir_ln_three_layers_rand = make_savedir(K, skip, q, map_estimate, lr)
print("Savedir for 0.01 LN Random estimation on [100, 50, 10]: ", savedir_ln_three_layers_rand)
perps, losses, zs, ws = train_and_test(K, savedir_ln_three_layers_rand, skip, q=q, lr=lr, n_iter_per_epoch=25000, 
	map_estimate=map_estimate, batch_size=-1, n_test_epoch=50)
print("Savedir for 0.01 LN Random estimation on [100, 50, 10]: ", savedir_ln_three_layers_rand)'''


## Section 1.5: Gamma vs LogNormal
## Section 1.5A: Gamma
'''K = [50, 25]
skip = False
map_estimate = False
q = 'gamma'
lr = 0.1
savedir_gamma_two_layers = make_savedir(K, skip, q, map_estimate, lr)
print("Savedir for 0.1 Gamma estimation on [50, 25]: ", savedir_gamma_two_layers)
perps, losses, zs, ws = train_and_test(K, savedir_gamma_two_layers, skip, q=q, lr=lr, n_iter_per_epoch=4000, 
	map_estimate=map_estimate, batch_size=-1, n_test_epoch=10)
print("Savedir for 0.1 Gamma estimation on [50, 25]: ", savedir_gamma_two_layers)'''
# Results stored in /info/20171129_002902_nips_50_25_gamma_lr_0.1

## Section 1.5B: LogNormal
'''K = [50, 25]
skip = False
map_estimate = False
q = 'lognormal'
lr = 0.1
savedir_ln_two_layers = make_savedir(K, skip, q, map_estimate, lr)
print("Savedir for 0.1 LN estimation on [50, 25]: ", savedir_ln_two_layers)
perps, losses, zs, ws = train_and_test(K, savedir_ln_two_layers, skip, q=q, lr=lr, n_iter_per_epoch=4000, 
	map_estimate=map_estimate, batch_size=-1, n_test_epoch=10)
print("Savedir for 0.1 LN estimation on [50, 25]: ", savedir_ln_two_layers)'''
# Results stored in /info/20171129_004346_nips_50_25_lognormal_lr_0.1
# /info/20171129_004346_nips_50_25_lognormal_lr_0.1

## Section 3
## Deep is unstable
'''K = [100, 100, 100, 50, 50, 50, 10, 10, 10]
skip = False
map_estimate = False
q = 'lognormal'
lr = 0.1
savedir_nine_layers_no_skips = make_savedir(K, skip, q, map_estimate, lr)
print("Savedir for 0.1 LN estimation on deepest model without skips: ", savedir_nine_layers_no_skips)
perps, losses, zs, ws = train_and_test(K, savedir_nine_layers_no_skips, skip, q=q, lr=lr, n_iter_per_epoch=500, 
	map_estimate=map_estimate, batch_size=-1, n_test_epoch=3)
print("Savedir for 0.1 LN estimation on deepest model without skips: ", savedir_nine_layers_no_skips)'''
# info/20171129_021335_nips_100_100_100_50_50_50_10_10_10_lognormal_lr_0.1


# 3B: Stable with skips
'''K = [100, 100, 100, 50, 50, 50, 10, 10, 10]
skip = True
map_estimate = False
q = 'lognormal'
lr = 0.1
savedir_nine_layers_with_skips = make_savedir(K, skip, q, map_estimate, lr)
print("Savedir for 0.1 LN estimation on deepest model with skips: ", savedir_nine_layers_with_skips)
perps, losses, zs, ws = train_and_test(K, savedir_nine_layers_with_skips, skip, q=q, lr=lr, n_iter_per_epoch=500, 
	map_estimate=map_estimate, batch_size=-1, n_test_epoch=2)
print("Savedir for 0.1 LN estimation on deepest model with skips: ", savedir_nine_layers_with_skips)'''
# info/20171129_021024_nips_100_100_100_50_50_50_10_10_10_lognormal_lr_0.1_skip

# Section 5

# Section 5A: No skips
'''K = [100, 100, 100, 50, 50, 50, 10, 10, 10]
skip = False
map_estimate = False
q = 'lognormal'
lr = 0.01
savedir_ln_nine_layers_none = make_savedir(K, skip, q, map_estimate, lr, most_skip=False)
print("Savedir for 9 layers none: ", savedir_ln_nine_layers_none)
perps, losses, zs, ws = train_and_test(K, savedir_ln_nine_layers_none, skip, q=q, lr=lr, n_iter_per_epoch=40000, 
    map_estimate=map_estimate, batch_size=-1, n_test_epoch=100, most_skip=False)
print("Savedir for 9 layers none: ", savedir_ln_nine_layers_none)'''


# Section 5B: Some skips
K = [100, 100, 100, 50, 50, 50, 10, 10, 10]
skip = True
map_estimate = False
q = 'lognormal'
lr = 0.01
savedir_ln_nine_layers_some = make_savedir(K, skip, q, map_estimate, lr, most_skip=False)
print("Savedir for 9 layers some: ", savedir_ln_nine_layers_some)
perps, losses, zs, ws = train_and_test(K, savedir_ln_nine_layers_some, skip, q=q, lr=lr, n_iter_per_epoch=40000, 
    map_estimate=map_estimate, batch_size=-1, n_test_epoch=100, most_skip=False)
print("Savedir for 9 layers some: ", savedir_ln_nine_layers_some)


# Section 5C: All skips
K = [100, 100, 100, 50, 50, 50, 10, 10, 10]
skip = False
map_estimate = False
q = 'lognormal'
lr = 0.01
savedir_ln_nine_layers_most = make_savedir(K, skip, q, map_estimate, lr, most_skip=True)
print("Savedir for 9 layers most: ", savedir_ln_nine_layers_most)
perps, losses, zs, ws = train_and_test(K, savedir_ln_nine_layers_most, skip, q=q, lr=lr, n_iter_per_epoch=40000, 
    map_estimate=map_estimate, batch_size=-1, n_test_epoch=100, most_skip=True)
print("Savedir for 9 layers most: ", savedir_ln_nine_layers_most)



