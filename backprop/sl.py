import jax
import jax.numpy as jnp  # JAX NumPy
from flax.training import train_state  # Useful dataclass to keep train state
import optax  # Optimizers
import tensorflow_datasets as tfds  # TFDS for MNIST
import wandb
import numpy as np

optimizer = {
    'adam': optax.adam,
    'sgd': optax.sgd
}
net = None


def get_datasets(dataset_name: str):
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder(dataset_name)
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    if 'id' in test_ds:
        test_ds.pop('id')
        train_ds.pop('id')
    return train_ds, test_ds


def create_train_state(rng, network, learning_rate, momentum):
    """Create initial `TrainState`."""
    global net
    net = network
    params = net.init(rng, jnp.ones(wandb.config.pholder), rng)['params']
    # schedule = optax.exponential_decay(learning_rate, 1, 0.999)
    tx = optimizer[wandb.config.opt_name](learning_rate, momentum)
    return train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx)

# def update_learning_rate(state, steps):
#     """Update learning rate based on the step."""
#     return state.replace(tx=state.tx.update_schedule(steps))

def update_train_state(learning_rate, momentum, params):
    """Update `TrainState`."""
    tx = optimizer[wandb.config.opt_name](learning_rate, momentum)
    return train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx)


def cross_entropy_loss(*, logits, labels, num_classes=10):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10) #TODO: fix this
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def compute_metrics(*, logits, labels, num_classes=10):
    loss = cross_entropy_loss(logits=logits, labels=labels, num_classes=num_classes)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy


@jax.jit
def eval_step(params, batch, rng):
    logits = net.apply({'params': params}, batch['image'], rng)
    return compute_metrics(logits=logits, labels=batch['label'])


def eval_model(params, test_ds, rng):
    rng, rng_net = jax.random.split(rng)
    loss, accuracy = eval_step(params, test_ds, rng_net)
    # metrics = jax.device_get(metrics)
    # summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
    # return summary['loss'], summary['accuracy']
    return loss, accuracy

@jax.jit
def train_step(state, X, y, rng):
    """Train for a single step."""

    def loss_fn(params):
        logits = net.apply({'params': params}, X, rng)
        loss = cross_entropy_loss(logits=logits, labels=y)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    loss, accuracy = compute_metrics(logits=logits, labels=y)
    return state, loss, accuracy

@jax.pmap
def train_step_pmap(state, X, y, rng):
    """Train for a single step."""

    def loss_fn(params):
        logits = net.apply({'params': params}, X, rng)
        loss = cross_entropy_loss(logits=logits, labels=y)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    loss, accuracy = compute_metrics(logits=logits, labels=y)
    return state, loss, accuracy

@jax.pmap
def train_step_pmap2(state, X, y, rng):
    return jax.vmap(train_step, in_axes=(0, 0, 0, None))(state, X, y, rng)

def train_single_epoch_pmap(state, X, y, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = X.shape[2]
    steps_per_epoch = train_ds_size // batch_size

    perm = jax.random.permutation(rng, train_ds_size)[batch_size]
    # perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    # perms = perms.reshape((steps_per_epoch, batch_size))
    batch_loss = []
    batch_acc = []
    rng, rng_net = jax.random.split(rng)
    # for perm in perms:
    X_batch = jnp.take(X, perm, axis=2)
    Y_batch = jnp.take(y, perm, axis=2)
    rngs = jax.random.split(rng_net, num=X.shape[0])
    state, loss, accuracy = train_step_pmap2(state, X_batch, Y_batch, rngs)
    batch_loss.append(loss)
    batch_acc.append(accuracy)

    return state, jnp.array(batch_loss).mean(), jnp.array(batch_acc).mean()


def train_epoch_pmap(state, X, y, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = X.shape[2]
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    rng, rng_net = jax.random.split(rng)

    # Rearrange your data once according to the permuted indices, instead of in every step
    X = jnp.take(X, perms, axis=2)
    Y = jnp.take(y, perms, axis=2)

    total_loss = 0
    total_acc = 0

    for i in range(steps_per_epoch):
        # Now you can simply slice your array instead of using jnp.take
        X_batch = X[:, :, i]
        Y_batch = Y[:, :, i]
        rngs = jax.random.split(rng_net, num=X.shape[0])
        state, loss, accuracy = train_step_pmap2(state, X_batch, Y_batch, rngs)

        # total_loss = jax.pmap(lambda x,y: x+y, in_axes=(None, 0))(total_loss, loss)
        # total_acc = jax.pmap(lambda x,y: x+y, in_axes=(None, 0))(total_acc, accuracy)

    # total_loss = total_loss / steps_per_epoch
    # total_acc = total_acc / steps_per_epoch

    return state, total_loss, total_acc


def train_single_batch(state, X, y, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(X)
    # steps_per_epoch = train_ds_size // batch_size

    perm = jax.random.permutation(rng, train_ds_size)[batch_size]
    # perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    # perms = perms.reshape((steps_per_epoch, batch_size))
    batch_loss = []
    batch_acc = []
    rng, rng_net = jax.random.split(rng)
    # for perm in perms:
    X_batch = jnp.take(X, perm, axis=0)
    Y_batch = jnp.take(y, perm, axis=0)
    state, loss, accuracy = train_step(state, X_batch, Y_batch, rng_net)
    batch_loss.append(loss)
    batch_acc.append(accuracy)

    return state, jnp.array(batch_loss).mean(), jnp.array(batch_acc).mean()

def train_epoch(state, X, y, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(X)
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Rearrange your data once according to the permuted indices, instead of in every step
    X = jnp.take(X, perms, axis=0)
    y = jnp.take(y, perms, axis=0)

    # batch_loss = []
    # batch_acc = []
    rng, rng_net = jax.random.split(rng)
    for i in range(steps_per_epoch):
        # Now you can simply slice your array instead of using jnp.take
        X_batch = X[i]
        Y_batch = y[i]
        state, loss, accuracy = train_step(state, X_batch, Y_batch, rng_net)
        # batch_loss.append(loss)
        # batch_acc.append(accuracy)

    return state, 0, 0 #jnp.array(batch_loss).mean(), jnp.array(batch_acc).mean()


def get_datasets_non_iid(dataset: str, n_clients: int):
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder(dataset)
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    train = [{'image': None, 'label': None} for i in range(n_clients)]
    # test_ = [{'image': None, 'label': None} for i in range(5)]
    for i in range(5):
        train[i]['image'] = jnp.concatenate((
            train_ds['image'][(train_ds['label'] == 2 * i + 0).nonzero()[0]],
            train_ds['image'][(train_ds['label'] == 2 * i + 1).nonzero()[0]]),
            axis=0)
        train[i]['label'] = jnp.concatenate((
            train_ds['label'][(train_ds['label'] == 2 * i + 0).nonzero()[0]],
            train_ds['label'][(train_ds['label'] == 2 * i + 1).nonzero()[0]]),
            axis=0)
    if 'id' in test_ds.keys():
        test_ds.pop('id')
    return train, test_ds
    # return train_ds, test_ds

def get_fed_datasets(dataset: str, n_clients: int, n_shards_per_client: int = 2, iid: bool = False):
    ds_builder = tfds.builder(dataset)
    ds_builder.download_and_prepare()

    # Load the entire dataset into memory
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    # Preprocess the images (normalize pixel values)
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.

    # Create the clients datasets
    train_clients = [None for _ in range(n_clients)]

    if iid:
        # Shuffle the data
        idx = np.random.permutation(len(train_ds['image']))
        n_classes = len(np.unique(train_ds['label']))
        train_ds['image'] = train_ds['image'][idx]
        train_ds['label'] = train_ds['label'][idx]

        # Assign samples to each client
        for i in range(n_clients):
            start_idx = i * n_shards_per_client * len(train_ds['image']) // n_clients // n_classes
            end_idx = (i + 1) * n_shards_per_client * len(train_ds['image']) // n_clients // n_classes
            train_clients[i] = {
                'image': train_ds['image'][start_idx:end_idx],
                'label': train_ds['label'][start_idx:end_idx]
            }
    else:
        # Compute shard size based on dataset size and number of clients
        shards = []
        n_classes = len(np.unique(train_ds['label']))
        for label in np.unique(train_ds['label']):
            label_indices = np.where(train_ds['label'] == label)[0]
            shard_size = len(label_indices) // n_shards_per_client if n_clients > n_classes else len(label_indices)
            label_indices = np.random.permutation(label_indices)

            shards.extend(np.array_split(label_indices, len(label_indices) // shard_size))

        # Assign shards to each client
        for i in range(n_clients):
            images = []
            labels = []
            for j in range(n_shards_per_client):
                # Make sure that we rotate through the shards to ensure that each client gets 2 different classes
                shard_indices = shards[(i + j * n_clients) % len(shards)]
                images.append(train_ds['image'][shard_indices])
                labels.append(train_ds['label'][shard_indices])

            train_clients[i] = {
                'image': np.concatenate(images, axis=0),
                'label': np.concatenate(labels, axis=0)
            }
    if 'id' in test_ds.keys():
        test_ds.pop('id')
    return train_clients, test_ds

from jax.lib import xla_bridge

def get_fed_datasets_pmap(dataset: str, n_clients: int, n_shards_per_client: int = 2, iid: bool = False):
    n_devices = xla_bridge.device_count()
    assert n_clients % n_devices == 0, "n_clients should be a multiple of n_devices for the data to be evenly distributed"

    ds_builder = tfds.builder(dataset)
    ds_builder.download_and_prepare()

    # Load the entire dataset into memory
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    # Preprocess the images (normalize pixel values)
    train_ds['image'] = np.float32(train_ds['image']) / 255.
    test_ds['image'] = np.float32(test_ds['image']) / 255.

    # Create the clients datasets
    train_clients = [None for _ in range(n_clients)]

    if iid:
        # Shuffle the data
        idx = np.random.permutation(len(train_ds['image']))
        train_ds['image'] = train_ds['image'][idx]
        train_ds['label'] = train_ds['label'][idx]

        # Assign samples to each client
        for i in range(n_clients):
            start_idx = i * n_shards_per_client * len(train_ds['image']) // n_clients
            end_idx = (i + 1) * n_shards_per_client * len(train_ds['image']) // n_clients
            train_clients[i] = {
                'image': train_ds['image'][start_idx:end_idx],
                'label': train_ds['label'][start_idx:end_idx]
            }
    else:
        # Compute shard size based on dataset size and number of clients
        shards = []
        n_classes = len(np.unique(train_ds['label']))
        for label in np.unique(train_ds['label']):
            label_indices = np.where(train_ds['label'] == label)[0]
            shard_size = len(label_indices) // n_shards_per_client if n_clients > n_classes else len(label_indices)
            label_indices = np.random.permutation(label_indices)

            shards.extend(np.array_split(label_indices, len(label_indices) // shard_size))

        # Assign shards to each client
        for i in range(n_clients):
            images = []
            labels = []
            for j in range(n_shards_per_client):
                # Make sure that we rotate through the shards to ensure that each client gets 2 different classes
                shard_indices = shards[(i + j * n_clients) % len(shards)]
                images.append(train_ds['image'][shard_indices])
                labels.append(train_ds['label'][shard_indices])

            train_clients[i] = {
                'image': np.concatenate(images, axis=0),
                'label': np.concatenate(labels, axis=0)
            }
        # Padding and reshaping
        max_samples = max([len(c['image']) for c in train_clients])
        for i in range(n_clients):
            n_samples = len(train_clients[i]['image'])
            if n_samples < max_samples:
                padding_images = jnp.zeros((max_samples - n_samples,) + train_ds['image'].shape[1:])
                padding_labels = jnp.zeros((max_samples - n_samples,) + train_ds['label'].shape[1:])
                train_clients[i]['image'] = jnp.concatenate([train_clients[i]['image'], padding_images])
                train_clients[i]['label'] = jnp.concatenate([train_clients[i]['label'], padding_labels])

        # Reshape data to [n_clients, n_samples, img_height, img_width, n_channels]
        train_clients = {k: np.stack([c[k] for c in train_clients]) for k in ['image', 'label']}
    if 'id' in test_ds.keys():
        test_ds.pop('id')
    return train_clients, test_ds

def get_datasets_iid(dataset: str, n_clients: int):
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder(dataset)
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    train = [{'image': None, 'label': None} for i in range(n_clients)]
    for i in range(5):
        train[i]['image'] = train_ds['image'][i * 10000:(i + 1) * 10000]
        train[i]['label'] = train_ds['label'][i * 10000:(i + 1) * 10000]
    if 'id' in test_ds.keys():
        test_ds.pop('id')
    return train, test_ds