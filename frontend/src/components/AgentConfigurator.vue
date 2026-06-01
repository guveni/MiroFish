<template>
  <div class="config-block">
    <div class="config-block-header">
      <span class="config-block-title">{{ $t('step2.agentConfig') }}</span>
      <span class="config-block-badge">{{ agentConfigs?.length || 0 }} {{ $t('common.items') }}</span>
    </div>
    <div class="agents-cards">
      <div 
        v-for="agent in agentConfigs" 
        :key="agent.agent_id" 
        class="agent-card"
      >
        <!-- 卡片头部 -->
        <div class="agent-card-header">
          <div class="agent-identity">
            <span class="agent-id">Agent {{ agent.agent_id }}</span>
            <span class="agent-name">{{ agent.entity_name }}</span>
          </div>
          <div class="agent-tags">
            <span class="agent-type">{{ agent.entity_type }}</span>
            <span class="agent-stance" :class="'stance-' + agent.stance">{{ agent.stance }}</span>
          </div>
        </div>
        
        <!-- 活跃时间轴 -->
        <div class="agent-timeline">
          <span class="timeline-label">{{ $t('step2.activeTimePeriod') }}</span>
          <div class="mini-timeline">
            <div 
              v-for="hour in 24" 
              :key="hour - 1" 
              class="timeline-hour"
              :class="{ 'active': agent.active_hours?.includes(hour - 1) }"
              :title="`${hour - 1}:00`"
            ></div>
          </div>
          <div class="timeline-marks">
            <span>0</span>
            <span>6</span>
            <span>12</span>
            <span>18</span>
            <span>24</span>
          </div>
        </div>

        <!-- 行为参数 -->
        <div class="agent-params">
          <div class="param-group">
            <div class="param-item">
              <span class="param-label">{{ $t('step2.postsPerHour') }}</span>
              <span class="param-value">{{ agent.posts_per_hour }}</span>
            </div>
            <div class="param-item">
              <span class="param-label">{{ $t('step2.commentsPerHour') }}</span>
              <span class="param-value">{{ agent.comments_per_hour }}</span>
            </div>
            <div class="param-item">
              <span class="param-label">{{ $t('step2.responseDelay') }}</span>
              <span class="param-value">{{ agent.response_delay_min }}-{{ agent.response_delay_max }}min</span>
            </div>
          </div>
          <div class="param-group">
            <div class="param-item">
              <span class="param-label">{{ $t('step2.activityLevel') }}</span>
              <span class="param-value with-bar">
                <span class="mini-bar" :style="{ width: (agent.activity_level * 100) + '%' }"></span>
                {{ (agent.activity_level * 100).toFixed(0) }}%
              </span>
            </div>
            <div class="param-item">
              <span class="param-label">{{ $t('step2.sentimentBias') }}</span>
              <span class="param-value" :class="agent.sentiment_bias > 0 ? 'positive' : agent.sentiment_bias < 0 ? 'negative' : 'neutral'">
                {{ agent.sentiment_bias > 0 ? '+' : '' }}{{ agent.sentiment_bias?.toFixed(1) }}
              </span>
            </div>
            <div class="param-item">
              <span class="param-label">{{ $t('step2.influenceWeight') }}</span>
              <span class="param-value highlight">{{ agent.influence_weight?.toFixed(1) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  agentConfigs: Array
})
</script>
