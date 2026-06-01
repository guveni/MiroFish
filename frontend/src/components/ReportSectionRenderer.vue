<template>
  <div v-if="reportOutline" class="report-content-wrapper">
    <!-- Sections List -->
    <div class="sections-list">
      <div 
        v-for="(section, idx) in reportOutline.sections" 
        :key="idx"
        class="report-section-item"
        :class="{ 
          'is-active': currentSectionIndex === idx + 1,
          'is-completed': isSectionCompleted(idx + 1),
          'is-pending': !isSectionCompleted(idx + 1) && currentSectionIndex !== idx + 1
        }"
      >
        <div class="section-header-row" @click="$emit('toggle-section-collapse', idx)" :class="{ 'clickable': isSectionCompleted(idx + 1) }">
          <span class="section-number">{{ String(idx + 1).padStart(2, '0') }}</span>
          <h3 class="section-title">{{ section.title }}</h3>
          <svg 
            v-if="isSectionCompleted(idx + 1)" 
            class="collapse-icon" 
            :class="{ 'is-collapsed': collapsedSections.has(idx) }"
            viewBox="0 0 24 24" 
            width="20" 
            height="20" 
            fill="none" 
            stroke="currentColor" 
            stroke-width="2"
          >
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </div>
        
        <div class="section-body" v-show="!collapsedSections.has(idx)">
          <!-- Completed Content -->
          <div v-if="generatedSections[idx + 1]" class="generated-content" v-html="renderMarkdown(generatedSections[idx + 1])"></div>
          
          <!-- Loading State -->
          <div v-else-if="currentSectionIndex === idx + 1" class="loading-state">
            <div class="loading-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <circle cx="12" cy="12" r="10" stroke-width="4" stroke="#E5E7EB"></circle>
                <path d="M12 2a10 10 0 0 1 10 10" stroke-width="4" stroke="#4B5563" stroke-linecap="round"></path>
              </svg>
            </div>
            <span class="loading-text">{{ $t('step4.generatingSection', { title: section.title }) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Waiting State -->
  <div v-else class="waiting-placeholder">
    <div class="waiting-animation">
      <div class="waiting-ring"></div>
      <div class="waiting-ring"></div>
      <div class="waiting-ring"></div>
    </div>
    <span class="waiting-text">Waiting for Report Agent...</span>
  </div>
</template>

<script setup>
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps({
  reportOutline: Object,
  currentSectionIndex: Number,
  generatedSections: Object,
  collapsedSections: Object
})

defineEmits(['toggle-section-collapse'])

const isSectionCompleted = (sectionIndex) => {
  return !!props.generatedSections[sectionIndex]
}

const renderMarkdown = (content) => {
  if (!content) return ''
  
  // Remove the heading at the start (## xxx) because section title is already shown outside
  let processedContent = content.replace(/^##\s+.+\n+/, '')
  
  // Process code blocks
  let html = processedContent.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="code-block"><code>$2</code></pre>')
  
  // Process inline code
  html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
  
  // Process headings
  html = html.replace(/^#### (.+)$/gm, '<h5 class="md-h5">$1</h5>')
  html = html.replace(/^### (.+)$/gm, '<h4 class="md-h4">$1</h4>')
  html = html.replace(/^## (.+)$/gm, '<h3 class="md-h3">$1</h3>')
  html = html.replace(/^# (.+)$/gm, '<h2 class="md-h2">$1</h2>')
  
  // Process blockquotes
  html = html.replace(/^> (.+)$/gm, '<blockquote class="md-quote">$1</blockquote>')
  
  // Process list - support sub-lists
  html = html.replace(/^(\s*)- (.+)$/gm, (match, indent, text) => {
    const level = Math.floor(indent.length / 2)
    return `<li class="md-li" data-level="${level}">${text}</li>`
  })
  
  // Process bold
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong class="md-strong">$1</strong>')
  
  // Process italic
  html = html.replace(/\*([^*]+)\*/g, '<em class="md-em">$1</em>')
  
  // Process link
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" class="md-link">$1</a>')
  
  // Process line breaks (non-empty lines not followed by headings or lists)
  const lines = html.split('\n')
  const processedLines = lines.map((line) => {
    const trimmed = line.trim()
    if (!trimmed) return '<div class="md-gap"></div>'
    if (trimmed.startsWith('<h') || trimmed.startsWith('<block') || trimmed.startsWith('<li') || trimmed.startsWith('<pre') || trimmed.startsWith('</pre')) {
      return line
    }
    return `<p class="md-p">${line}</p>`
  })
  
  html = processedLines.join('\n')
  
  // Process source citation markers: [S1], [S2] etc
  html = html.replace(/\[[Ss](\d+)\]/g, '<span class="source-badge">S$1</span>')
  
  // Process adversarial / challenge markers
  html = html.replace(/<self_critique>/gi, '<div class="self-critique-block"><div class="critique-header"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01"/></svg><span>Self-Critique & Calibration</span></div><div class="critique-body">')
  html = html.replace(/<\/self_critique>/gi, '</div></div>')
  
  return html
}
</script>
