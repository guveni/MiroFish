<template>
  <div class="report-meta">
    <div class="meta-left">
      <span class="report-tag">Prediction Report</span>
      <span class="report-id">ID: {{ reportId || 'REF-2024-X92' }}</span>
    </div>
    <div class="report-actions">
      <button class="action-btn" @click="$emit('toggle-focus-mode')" :title="isFocusMode ? $t('step4.exitFocus') : $t('step4.focusMode')">
        <svg v-if="isFocusMode" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M4 14h6v6m10-6h-6v6M4 10h6V4m10 6h-6V4"/>
        </svg>
        <svg v-else viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/>
        </svg>
        <span>{{ isFocusMode ? $t('step4.exitFocus') : $t('step4.focusMode') }}</span>
      </button>
      <button class="action-btn" @click="$emit('download-pdf')" :title="$t('step4.downloadPDF')">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>
        </svg>
        <span>{{ $t('step4.downloadPDF') }}</span>
      </button>
      <button class="action-btn" @click="$emit('recreate-report')" :disabled="isRecreating || !simulationId" :title="$t('step4.recreateReport')">
        <svg v-if="isRecreating" class="spin-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" stroke-width="3" stroke="#E5E7EB"></circle>
          <path d="M12 2a10 10 0 0 1 10 10" stroke-width="3" stroke="#4B5563" stroke-linecap="round"></path>
        </svg>
        <svg v-else viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
        </svg>
        <span>{{ isRecreating ? $t('step4.recreating') : $t('step4.recreateReport') }}</span>
      </button>
    </div>
  </div>
</template>

<script setup>
defineProps({
  reportId: String,
  isFocusMode: Boolean,
  isRecreating: Boolean,
  simulationId: String
})

defineEmits(['toggle-focus-mode', 'download-pdf', 'recreate-report'])
</script>
