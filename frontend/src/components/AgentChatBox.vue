<template>
  <div class="chat-container">
    <!-- Report Agent System Info Tools -->
    <div v-if="chatTarget === 'report_agent'" class="report-agent-tools-card">
      <div class="tools-card-header">
        <div class="tools-card-avatar">R</div>
        <div class="tools-card-info">
          <div class="tools-card-name">{{ $t('step5.reportAgentChat') }}</div>
          <div class="tools-card-subtitle">{{ $t('step5.reportAgentDesc') }}</div>
        </div>
        <button class="tools-card-toggle" @click="showToolsDetail = !showToolsDetail">
          <svg :class="{ 'is-expanded': showToolsDetail }" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </button>
      </div>
      <div v-if="showToolsDetail" class="tools-card-body">
        <div class="tools-grid">
          <div class="tool-item tool-purple">
            <div class="tool-icon-wrapper">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.5V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.5A7 7 0 0 0 12 2z"></path>
              </svg>
            </div>
            <div class="tool-content">
              <div class="tool-name">{{ $t('step5.toolInsightForge') }}</div>
              <div class="tool-desc">{{ $t('step5.toolInsightForgeDesc') }}</div>
            </div>
          </div>
          <div class="tool-item tool-blue">
            <div class="tool-icon-wrapper">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
              </svg>
            </div>
            <div class="tool-content">
              <div class="tool-name">{{ $t('step5.toolPanoramaSearch') }}</div>
              <div class="tool-desc">{{ $t('step5.toolPanoramaSearchDesc') }}</div>
            </div>
          </div>
          <div class="tool-item tool-orange">
            <div class="tool-icon-wrapper">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
              </svg>
            </div>
            <div class="tool-content">
              <div class="tool-name">{{ $t('step5.toolQuickSearch') }}</div>
              <div class="tool-desc">{{ $t('step5.toolQuickSearchDesc') }}</div>
            </div>
          </div>
          <div class="tool-item tool-green">
            <div class="tool-icon-wrapper">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                <circle cx="9" cy="7" r="4"></circle>
                <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"></path>
              </svg>
            </div>
            <div class="tool-content">
              <div class="tool-name">{{ $t('step5.toolInterviewSubAgent') }}</div>
              <div class="tool-desc">{{ $t('step5.toolInterviewSubAgentDesc') }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Agent Profile Card -->
    <div v-if="chatTarget === 'agent' && selectedAgent" class="agent-profile-card">
      <div class="profile-card-header">
        <div class="profile-card-avatar">{{ (selectedAgent.username || 'A')[0] }}</div>
        <div class="profile-card-info">
          <div class="profile-card-name">{{ selectedAgent.username }}</div>
          <div class="profile-card-meta">
            <span v-if="selectedAgent.name" class="profile-card-handle">@{{ selectedAgent.name }}</span>
            <span class="profile-card-profession">{{ selectedAgent.profession || $t('step2.unknownProfession') }}</span>
          </div>
        </div>
        <button class="profile-card-toggle" @click="showFullProfile = !showFullProfile">
          <svg :class="{ 'is-expanded': showFullProfile }" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </button>
      </div>
      <div v-if="showFullProfile && selectedAgent.bio" class="profile-card-body">
        <div class="profile-card-bio">
          <div class="profile-card-label">{{ $t('step5.profileBio') }}</div>
          <p>{{ selectedAgent.bio }}</p>
        </div>
      </div>
    </div>

    <!-- Chat Messages -->
    <div class="chat-messages" ref="chatMessages">
      <div v-if="chatHistory.length === 0" class="chat-empty">
        <div class="empty-icon">
          <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        </div>
        <p class="empty-text">
          {{ chatTarget === 'report_agent' ? $t('step5.chatEmptyReportAgent') : $t('step5.chatEmptyAgent') }}
        </p>
      </div>
      <div 
        v-for="(msg, idx) in chatHistory" 
        :key="idx"
        class="chat-message"
        :class="msg.role"
      >
        <div class="message-avatar">
          <span v-if="msg.role === 'user'">U</span>
          <span v-else>{{ msg.role === 'assistant' && chatTarget === 'report_agent' ? 'R' : (selectedAgent?.username?.[0] || 'A') }}</span>
        </div>
        <div class="message-content">
          <div class="message-header">
            <span class="sender-name">
              {{ msg.role === 'user' ? 'You' : (chatTarget === 'report_agent' ? 'Report Agent' : (selectedAgent?.username || 'Agent')) }}
            </span>
            <span class="message-time">{{ formatTime(msg.timestamp) }}</span>
          </div>
          <div class="message-text" v-html="renderMarkdown(msg.content)"></div>
        </div>
      </div>
      <div v-if="isSending" class="chat-message assistant">
        <div class="message-avatar">
          <span>{{ chatTarget === 'report_agent' ? 'R' : (selectedAgent?.username?.[0] || 'A') }}</span>
        </div>
        <div class="message-content">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
    </div>

    <!-- Chat Input -->
    <div class="chat-input-area">
      <textarea 
        v-model="chatInput"
        class="chat-input"
        :placeholder="$t('step5.chatInputPlaceholder')"
        @keydown.enter.exact.prevent="handleSend"
        :disabled="isSending || (!selectedAgent && chatTarget === 'agent')"
        rows="1"
        ref="chatInputRef"
      ></textarea>
      <button 
        class="send-btn"
        @click="handleSend"
        :disabled="!chatInput.trim() || isSending || (!selectedAgent && chatTarget === 'agent')"
      >
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({
  chatTarget: String,
  selectedAgent: Object,
  chatHistory: Array,
  isSending: Boolean,
  formatTime: Function,
  renderMarkdown: Function
})

const emit = defineEmits(['send-message'])

const chatInput = ref('')
const showToolsDetail = ref(true)
const showFullProfile = ref(false)
const chatMessages = ref(null)
const chatInputRef = ref(null)

const handleSend = () => {
  if (!chatInput.value.trim() || props.isSending) return
  emit('send-message', chatInput.value.trim())
  chatInput.value = ''
}

// Scroll to bottom helper
const scrollToBottom = () => {
  nextTick(() => {
    if (chatMessages.value) {
      chatMessages.value.scrollTop = chatMessages.value.scrollHeight
    }
  })
}

// Watch chatHistory length or isSending to scroll
watch(() => props.chatHistory.length, scrollToBottom)
watch(() => props.isSending, scrollToBottom)
</script>
