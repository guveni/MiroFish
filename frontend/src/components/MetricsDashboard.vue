<template>
  <div class="survey-container">
    <!-- Survey Setup -->
    <div class="survey-setup">
      <div class="setup-section">
        <div class="section-header">
          <span class="section-title">{{ $t('step5.selectSurveyTarget') }}</span>
          <span class="selection-count">{{ $t('step5.selectedCount', { selected: selectedAgents.size, total: profiles.length }) }}</span>
        </div>
        <div class="agents-grid">
          <label 
            v-for="(agent, idx) in profiles" 
            :key="idx"
            class="agent-checkbox"
            :class="{ checked: selectedAgents.has(idx) }"
          >
            <input 
              type="checkbox" 
              :checked="selectedAgents.has(idx)"
              @change="$emit('toggle-agent-selection', idx)"
            >
            <div class="checkbox-avatar">{{ (agent.username || 'A')[0] }}</div>
            <div class="checkbox-info">
              <span class="checkbox-name">{{ agent.username }}</span>
              <span class="checkbox-role">{{ agent.profession || $t('step2.unknownProfession') }}</span>
            </div>
            <div class="checkbox-indicator">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="3">
                <polyline points="20 6 9 17 4 12"></polyline>
              </svg>
            </div>
          </label>
        </div>
        <div class="selection-actions">
          <button class="action-link" @click="$emit('select-all-agents')">{{ $t('step5.selectAll') }}</button>
          <span class="action-divider">|</span>
          <button class="action-link" @click="$emit('clear-agent-selection')">{{ $t('step5.clearSelection') }}</button>
        </div>
      </div>

      <div class="setup-section">
        <div class="section-header">
          <span class="section-title">{{ $t('step5.surveyQuestions') }}</span>
        </div>
        <textarea 
          v-model="surveyInputText"
          class="survey-input"
          :placeholder="$t('step5.surveyInputPlaceholder')"
          rows="3"
        ></textarea>
      </div>

      <button 
        class="survey-submit-btn"
        :disabled="selectedAgents.size === 0 || !surveyInputText.trim() || isSurveying"
        @click="handleSubmit"
      >
        <span v-if="isSurveying" class="loading-spinner"></span>
        <span v-else>{{ $t('step5.submitSurvey') }}</span>
      </button>
    </div>

    <!-- Survey Results -->
    <div v-if="surveyResults.length > 0" class="survey-results">
      <div class="results-header">
        <span class="results-title">{{ $t('step5.surveyResults') }}</span>
        <span class="results-count">{{ $t('step5.surveyResultsCount', { count: surveyResults.length }) }}</span>
      </div>
      <div class="results-list">
        <div 
          v-for="(result, idx) in surveyResults" 
          :key="idx"
          class="result-card"
        >
          <div class="result-header">
            <div class="result-avatar">{{ (result.agent_name || 'A')[0] }}</div>
            <div class="result-info">
              <span class="result-name">{{ result.agent_name }}</span>
              <span class="result-role">{{ result.profession || $t('step2.unknownProfession') }}</span>
            </div>
          </div>
          <div class="result-question">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"></circle>
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            <span>{{ result.question }}</span>
          </div>
          <div class="result-answer" v-html="renderMarkdown(result.answer)"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  selectedAgents: Object,
  profiles: Array,
  isSurveying: Boolean,
  surveyResults: Array,
  renderMarkdown: Function
})

const emit = defineEmits([
  'toggle-agent-selection',
  'select-all-agents',
  'clear-agent-selection',
  'submit-survey'
])

const surveyInputText = ref('')

const handleSubmit = () => {
  if (props.selectedAgents.size === 0 || !surveyInputText.value.trim() || props.isSurveying) return
  emit('submit-survey', surveyInputText.value.trim())
  surveyInputText.value = ''
}
</script>
