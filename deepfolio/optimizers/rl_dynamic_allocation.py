import tensorflow as tf
import numpy as np

class MarketEnvironment:
    def __init__(self, returns, initial_balance=10000, transaction_cost=0.001):
        self.returns = returns
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.position = np.zeros(self.returns.shape[1])
        self.time = 0
        return self._get_state()
    
    def step(self, action):
        old_position = self.position
        self.position = action
        
        # Apply transaction costs
        self.balance -= np.sum(np.abs(self.position - old_position)) * self.balance * self.transaction_cost
        
        # Apply market returns
        self.balance *= 1 + np.sum(self.position * self.returns[self.time])
        
        self.time += 1
        done = self.time >= len(self.returns)
        
        return self._get_state(), self._get_reward(), done
    
    def _get_state(self):
        return np.concatenate([
            self.position,
            [self.balance],
            self.returns[self.time] if self.time < len(self.returns) else np.zeros_like(self.returns[0])
        ])
    
    def _get_reward(self):
        return np.log(self.balance / self.initial_balance)

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='softmax')
        ])
    
    def call(self, state):
        return self.model(state)

class Critic(tf.keras.Model):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    def call(self, state):
        return self.model(state)

class RLDynamicAllocation(tf.keras.Model):
    def __init__(self, state_dim, action_dim, lr_actor=0.0001, lr_critic=0.001):
        super(RLDynamicAllocation, self).__init__()
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr_critic)
    
    def train(self, env, episodes=1000):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
                    action_probs = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
                    action = tf.random.categorical(tf.math.log(action_probs), 1)[0, 0]
                    action_onehot = tf.one_hot(action, env.action_space.n)
                    
                    next_state, reward, done = env.step(action_onehot.numpy())
                    
                    critic_value = self.critic(tf.convert_to_tensor([state], dtype=tf.float32))
                    next_critic_value = self.critic(tf.convert_to_tensor([next_state], dtype=tf.float32))
                    
                    advantage = reward + 0.99 * next_critic_value * (1 - done) - critic_value
                    actor_loss = -tf.math.log(action_probs[0, action]) * advantage
                    critic_loss = advantage ** 2
                
                actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
                
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                
                state = next_state
    
    def get_action(self, state):
        action_probs = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
        return action_probs.numpy()[0]