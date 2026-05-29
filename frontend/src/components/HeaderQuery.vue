<template>
  <div class="header-query-wrapper" @mouseenter="showTooltip = true" @mouseleave="showTooltip = false">
    <div class="header-query">
      <span class="query-label">Query:</span>
      <span class="query-text">{{ text }}</span>
    </div>
    <div v-if="showTooltip && text" class="query-tooltip">
      <div class="tooltip-label">Query:</div>
      <div class="tooltip-content">{{ text }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  text: {
    type: String,
    default: ''
  }
})

const showTooltip = ref(false)
</script>

<style scoped>
.header-query-wrapper {
  position: relative;
  display: flex;
  min-width: 0; /* allows flex children to shrink below content size */
  flex: 1;
  height: 100%;
  align-items: center;
}

.header-query {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  overflow: hidden;
  color: #666;
  cursor: default;
  width: 100%;
}

.query-label {
  font-weight: 600;
  color: #000;
  flex-shrink: 0;
}

.query-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.query-tooltip {
  position: absolute;
  top: 45px;
  left: 0;
  width: max-content;
  max-width: 400px;
  max-height: 400px;
  overflow-y: auto;
  background: #FFF;
  border: 1px solid #EAEAEA;
  padding: 12px;
  border-radius: 6px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
  z-index: 1000;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tooltip-label {
  font-weight: 600;
  font-size: 12px;
  color: #999;
}

.tooltip-content {
  font-size: 13px;
  line-height: 1.5;
  color: #333;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
