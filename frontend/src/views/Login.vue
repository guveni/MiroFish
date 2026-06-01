<template>
  <div class="login-container">
    <div class="login-card">
      <!-- Left side: Hero branding -->
      <div class="login-brand-side">
        <div class="brand-header">
          <span class="brand-dot">■</span> MIROFISH
        </div>
        <div class="brand-hero">
          <h1 class="brand-title">Swarm-Intelligence Prediction Engine</h1>
          <p class="brand-desc">
            Simulate collective intelligence and run decentralized multi-agent scenarios to forecast real-world social dynamics.
          </p>
        </div>
        <div class="brand-footer">
          <span class="tech-tag">DECENTRALIZED AGENTS</span>
          <span class="tech-tag">COGNITIVE SIMULATION</span>
        </div>
      </div>

      <!-- Right side: Sign-in form -->
      <div class="login-form-side">
        <div class="form-header">
          <h2>Welcome to MiroFish</h2>
          <p class="subtitle">Please sign in to access your custom workspace</p>
        </div>

        <form @submit.prevent="handleLogin" class="auth-form">
          <div class="form-group">
            <label for="username">User Name</label>
            <div class="input-wrapper">
              <span class="input-icon">👤</span>
              <input 
                id="username" 
                v-model="form.name" 
                type="text" 
                placeholder="Enter your name" 
                required
              />
            </div>
          </div>

          <div class="form-group">
            <label for="email">Email Address</label>
            <div class="input-wrapper">
              <span class="input-icon">✉️</span>
              <input 
                id="email" 
                v-model="form.email" 
                type="email" 
                placeholder="name@example.com" 
                required
              />
            </div>
          </div>

          <button type="submit" class="submit-btn" :disabled="loading">
            <span v-if="loading">Signing in...</span>
            <span v-else>Enter Workspace <span class="arrow">→</span></span>
          </button>
        </form>

        <div class="divider">
          <span>OR QUICK SIGN-IN PRESETS</span>
        </div>

        <div class="presets-grid">
          <button 
            v-for="preset in presets" 
            :key="preset.email" 
            class="preset-card"
            @click="selectPreset(preset)"
          >
            <div class="preset-avatar">{{ preset.avatar }}</div>
            <div class="preset-info">
              <div class="preset-name">{{ preset.name }}</div>
              <div class="preset-email">{{ preset.email }}</div>
            </div>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const loading = ref(false)

const form = reactive({
  name: '',
  email: ''
})

const presets = [
  { name: 'Lead Predictor', email: 'lead@mirofish.ai', avatar: '🎯' },
  { name: 'Swarm Specialist', email: 'swarm@mirofish.ai', avatar: '🐝' },
  { name: 'Strategic Analyst', email: 'analyst@mirofish.ai', avatar: '📊' }
]

function selectPreset(preset) {
  form.name = preset.name
  form.email = preset.email
  handleLogin()
}

function handleLogin() {
  if (!form.name || !form.email) return
  
  loading.value = true
  
  // Create a dummy token for local auth: dummy_usr_<name>_<email>
  const normalizedName = form.name.replace(/\s+/g, '')
  const token = `dummy_usr_${normalizedName}_${form.email}`
  
  // Store user info and token in localStorage
  localStorage.setItem('mirofish_token', token)
  localStorage.setItem('mirofish_user', JSON.stringify({
    name: form.name,
    email: form.email
  }))
  
  setTimeout(() => {
    loading.value = false
    router.push('/')
  }, 600)
}
</script>

<style scoped>
.login-container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background-color: #0b0d10;
  font-family: 'Inter', sans-serif;
  color: #e2e8f0;
  padding: 20px;
}

.login-card {
  display: flex;
  width: 1000px;
  max-width: 100%;
  min-height: 600px;
  background-color: #12151a;
  border: 1px solid #1f242c;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
}

/* Left side styling */
.login-brand-side {
  flex: 1;
  background: linear-gradient(135deg, #181d24 0%, #0d1116 100%);
  border-right: 1px solid #1f242c;
  padding: 40px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  position: relative;
}

.login-brand-side::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-image: radial-gradient(#ef6820 1px, transparent 1px);
  background-size: 24px 24px;
  opacity: 0.05;
}

.brand-header {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 800;
  font-size: 1.25rem;
  letter-spacing: 2px;
  color: #ef6820;
  display: flex;
  align-items: center;
  gap: 8px;
}

.brand-dot {
  font-size: 0.8rem;
  color: #ef6820;
}

.brand-hero {
  margin-bottom: 40px;
  z-index: 1;
}

.brand-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.25rem;
  font-weight: 700;
  line-height: 1.2;
  color: #f8fafc;
  margin-bottom: 16px;
}

.brand-desc {
  font-size: 1rem;
  line-height: 1.6;
  color: #94a3b8;
}

.brand-footer {
  display: flex;
  gap: 12px;
  z-index: 1;
}

.tech-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  background-color: rgba(239, 104, 32, 0.08);
  border: 1px solid rgba(239, 104, 32, 0.2);
  color: #ef6820;
  padding: 4px 10px;
  border-radius: 4px;
}

/* Right side styling */
.login-form-side {
  flex: 1.2;
  padding: 40px 50px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.form-header {
  margin-bottom: 32px;
}

.form-header h2 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.75rem;
  font-weight: 600;
  color: #f8fafc;
  margin: 0 0 8px 0;
}

.subtitle {
  font-size: 0.925rem;
  color: #64748b;
  margin: 0;
}

.auth-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-group label {
  font-size: 0.85rem;
  font-weight: 500;
  color: #94a3b8;
}

.input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.input-icon {
  position: absolute;
  left: 14px;
  font-size: 0.95rem;
  color: #64748b;
}

.input-wrapper input {
  width: 100%;
  background-color: #1a1f26;
  border: 1px solid #2d3643;
  color: #f8fafc;
  padding: 12px 16px 12px 42px;
  border-radius: 6px;
  font-size: 0.95rem;
  transition: all 0.2s ease;
}

.input-wrapper input:focus {
  outline: none;
  border-color: #ef6820;
  box-shadow: 0 0 0 2px rgba(239, 104, 32, 0.15);
  background-color: #1e252e;
}

.submit-btn {
  background-color: #ef6820;
  color: #ffffff;
  border: none;
  padding: 14px;
  border-radius: 6px;
  font-weight: 600;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-top: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.submit-btn:hover {
  background-color: #f07432;
  transform: translateY(-1px);
}

.submit-btn:active {
  transform: translateY(0);
}

.submit-btn:disabled {
  background-color: #4b230d;
  color: #94a3b8;
  cursor: not-allowed;
}

.arrow {
  transition: transform 0.2s ease;
}

.submit-btn:hover .arrow {
  transform: translateX(3px);
}

.divider {
  display: flex;
  align-items: center;
  text-align: center;
  margin: 32px 0 24px 0;
}

.divider::before,
.divider::after {
  content: '';
  flex: 1;
  border-bottom: 1px solid #1f242c;
}

.divider span {
  padding: 0 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.725rem;
  color: #475569;
  letter-spacing: 1px;
}

.presets-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.preset-card {
  background-color: #171c24;
  border: 1px solid #232a35;
  border-radius: 8px;
  padding: 12px 10px;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  transition: all 0.2s ease;
  text-align: center;
}

.preset-card:hover {
  border-color: rgba(239, 104, 32, 0.4);
  background-color: #1b222c;
  transform: translateY(-2px);
}

.preset-avatar {
  font-size: 1.5rem;
}

.preset-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  width: 100%;
}

.preset-name {
  font-size: 0.8rem;
  font-weight: 600;
  color: #f1f5f9;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.preset-email {
  font-size: 0.675rem;
  color: #64748b;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Responsive adjustments */
@media (max-width: 850px) {
  .login-card {
    flex-direction: column;
    min-height: auto;
    width: 480px;
  }
  
  .login-brand-side {
    padding: 30px;
    border-right: none;
    border-bottom: 1px solid #1f242c;
  }
  
  .brand-title {
    font-size: 1.75rem;
  }
  
  .login-form-side {
    padding: 30px;
  }
}
</style>
