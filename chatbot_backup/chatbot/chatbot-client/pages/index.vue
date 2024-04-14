<script setup>
definePageMeta({
  middleware: ["auth"],
  path: '/:id?',
  keepalive: true
})

const { $i18n } = useNuxtApp()
const runtimeConfig = useRuntimeConfig()
const drawer = useDrawer()
const route = useRoute()
const conversation = ref(getDefaultConversationData())

const loadConversation = async () => {
  const { data, error } = await useAuthFetch('/api/chat/conversations/' + route.params.id)
  if (!error.value) {
    conversation.value = Object.assign(conversation.value, data.value)
  }
}

const loadMessage = async () => {
  const { data, error } = await useAuthFetch('/api/chat/messages/?conversationId=' + route.params.id)
  if (!error.value) {
    conversation.value.messages = data.value
    conversation.value.id = route.params.id
  }
}

const createNewConversation = () => {
  if (route.path !== '/') {
    return navigateTo('/?new')
  }
  conversation.value = Object.assign(getDefaultConversationData(), {
    topic: $i18n.t('newConversation')
  })
}


onMounted(async () => {
  if (route.params.id) {
    conversation.value.loadingMessages = true
    await loadConversation()
    await loadMessage()
    conversation.value.loadingMessages = false
  }
})


const navTitle = computed(() => {
  if (conversation.value && conversation.value.topic !== null) {
    return conversation.value.topic === '' ? $i18n.t('defaultConversationTitle') : conversation.value.topic
  }
  return runtimeConfig.public.appName
})

onActivated(async () => {
  if (route.path === '/' && route.query.new !== undefined) {
    createNewConversation()
  }
})

const redirectToMonitoringSystem = async () => {
  try {
    // 发送请求到后端，并获取响应
    const response = await fetch('/api/monitor/login', {
      method: 'GET', 
      headers: {
        'Accept': 'text/html',
      },
    });
    window.location.href = 'https://3dprinter.quest/api/monitor/login';
  } catch (error) {
    console.error('Error during redirection to the monitoring system:', error);
  }
};

</script>

<template>
  <v-app-bar>
    <v-app-bar-nav-icon @click="drawer = !drawer"></v-app-bar-nav-icon>

    <v-toolbar-title>{{ navTitle }}</v-toolbar-title>

    <v-spacer></v-spacer>


    <v-btn
      :title="$t('monitoringsystem')"
      @click="redirectToMonitoringSystem"
      class="d-md-none ma-3"
      style="display: flex; justify-content: center; align-items: center;"
    >
      <v-icon style="transform: translateX(25px);">monitoring</v-icon>
    </v-btn>
    
    <v-btn
      variant="outlined"
      class="text-none d-none d-md-block"
      @click="redirectToMonitoringSystem"
      style="margin-right: 50px;"
    >
      {{ $t('monitoringsystem') }}
    </v-btn>

    <v-btn
        :title="$t('newConversation')"
        icon="add"
        @click="createNewConversation"
        class="d-md-none ma-3"
    ></v-btn>

    <v-btn
        variant="outlined"
        class="text-none d-none d-md-block"
        @click="createNewConversation"
    >
      {{ $t('newConversation') }}
    </v-btn>

  </v-app-bar>

  <v-main>
    <Welcome v-if="!route.params.id && conversation.messages.length === 0" />
    <Conversation :conversation="conversation" />
  </v-main>
</template>