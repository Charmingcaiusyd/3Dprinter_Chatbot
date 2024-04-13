<script setup>
import { isMobile } from 'is-mobile'
const { $i18n } = useNuxtApp()

const props = defineProps({
  sendMessage: {
    type: Function,
    required: true
  },
  disabled: {
    type: Boolean,
    default: false
  },
  loading: {
    type: Boolean,
    default: false
  }
})

const message = ref('')
const rows = ref(1)
const autoGrow = ref(true)

const hint = computed(() => {
  return isMobile() ? '' : $i18n.t('pressEnterToSendYourMessageOrShiftEnterToAddANewLine')
})

watchEffect(() => {
  const lines = message.value.split(/\r\n|\r|\n/).length
  if (lines > 8) {
    rows.value = 8
    autoGrow.value = false
  } else {
    rows.value = 1
    autoGrow.value = true
  }
})

const file = ref(''); // 存储文件数据的响应式变量
const imageUrl = ref(''); // 存储图片预览URL的响应式变量

// 当点击回形针图标时触发的方法
const upload = () => {
  // 触发文件输入
  let fileInput = document.querySelector('input[type=file]');
  fileInput.click();
};

// 处理文件上传，并发送到服务器
const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('key', '3c20e134a78ba35b50dfe856ad04d7e1'); // Use your actual API key

  try {
    const response = await fetch('https://api.imgbb.com/1/upload', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    if (data.success) {
      imageUrl.value = data.data.url; // Set the image URL
      console.log('Image uploaded successfully: ', imageUrl.value); // Debug message
    } else {
      throw new Error('Image upload failed');
    }
  } catch (error) {
    console.error('Error during image upload:', error);
  }
};

const send = () => {
  let msg = message.value.trim();

  // remove the last "\n" if it exists
  if (msg.endsWith("\n")) {
    msg = msg.slice(0, -1);
  }

  if (msg.length > 0 || imageUrl.value) {
    let item = toolSelector.value.list[toolSelector.value.selected];
    let payload = {
      content: msg,
      url: imageUrl.value,
      tool: item.name,
      message_type: item.type
    };

    // If imageUrl is available, append it to the payload content in Markdown format
    if (imageUrl.value) {
      payload.content += `\n\n![Image](${imageUrl.value})`; // Markdown format for images
    }

    // Proceed with sending the message
    console.log("Sending message with payload:", payload);
    props.sendMessage(payload);

    // Clearing after sending
    message.value = "";
    imageUrl.value = "";
  } else {
    console.log("Attempted to send an empty message.");
  }

  console.log("Send function triggered"); // Debug message
  // Existing send logic...
  // 注释掉了 与上面的功能冲突
  //fetchReply(message);
  //console.log("fetchReply called with message:", message); // Debug message

  // Clearing after sending
  message.value = "";
  imageUrl.value = "";
  toolSelector.value.selected = 0;
};

const handleFileSelect = async () => {
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/jpeg, image/png';
  fileInput.onchange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      await uploadImage(file); // Use the uploadImage function to handle the upload
    }
  };
  fileInput.click();
};


const textArea = ref()
const documentMan = ref(null)

const usePrompt = (prompt) => {
  message.value = prompt
  textArea.value.focus()
}
const refreshDocList = () => {
  documentMan.value.loadDocs()
}

const clickSendBtn = () => {
  send()
}

const enterOnly = (event) => {
  event.preventDefault();
  if (!isMobile()) {
    send()
  }
}

defineExpose({
  usePrompt, refreshDocList
})

const toolSelector = ref({
  list: [
    { title: "Chat", icon: "add", name: "chat", type: 0 },
//    { title: "Web Search", icon: "travel_explore", name: "web_search", type: 100 },
//    { title: "ArXiv", icon: "local_library", name: "arxiv", type: 110 },
  ],
  selected: 0,
})
function getToolIcon() {
  let v = toolSelector.value
  let icon = v.list[v.selected].icon
  return icon
}
function getLabel() {
  let v = toolSelector.value
  let name = v.list[v.selected].name
  return "messageLabel." + name
}
function selectTool(idx) {
  let v = toolSelector.value
  v.selected = idx
  let name = v.list[idx].name
}
const docDialogCtl = ref({
  dialog: false,
})
</script>

<template>
  <div
      class="flex-grow-1 d-flex align-center justify-space-between"
  >
    <!-- <v-btn
      title="Tools"
      :icon="getToolIcon()"
      density="compact"
      size="default"
      class="mr-3"
      id="tools_btn"
    >
    </v-btn> -->
    <!-- <v-menu
      activator="#tools_btn"
      open-on-hover
    >
      <v-list density="compact">
        <v-list-item
          v-for="(item, index) in toolSelector.list"
          :key="index"
          :prepend-icon="item.icon"
          @click="selectTool(index)"
        >
          <v-list-item-title>{{ item.title }}</v-list-item-title>
        </v-list-item>
        <v-list-item
          prepend-icon="article"
          @click="docDialogCtl.dialog = true"
        >
          Documents
        </v-list-item>
      </v-list>
    </v-menu> -->
    <v-textarea
        ref="textArea"
        v-model="message"
        :label="$t(getLabel())"
        :placeholder="hint"
        :rows="rows"
        max-rows="8"
        :auto-grow="autoGrow"
        :disabled="disabled"
        :loading="loading"
        :hide-details="true"
        clearable
        variant="outlined"
        class="userinputmsg"
        @keydown.enter.exact="enterOnly"
    ></v-textarea>


    <!-- 文件上传按钮，图标为回形针 -->
    <v-btn icon @click="upload">
      <v-icon>attach_file</v-icon>
    </v-btn>
    <v-btn icon @click="handleFileSelect">
      <v-icon>file_upload</v-icon>
    </v-btn>
    <!-- 隐藏的文件输入，接受图片文件 -->
    <v-file-input
      v-model="file"
      show-size
      class="d-none"
      @change="handleFileUpload"
      accept="image/jpeg,image/png"
    ></v-file-input>
    
    <v-btn
        :disabled="loading"
        icon="send"
        title="Send"
        class="ml-3"
        @click="clickSendBtn"
    ></v-btn>
  </div>
  <img v-if="imageUrl" :src="imageUrl" alt="Uploaded image preview" style="max-width: 80px; max-height: 80px;" />
  <DocumentsManage
    :send-message="sendMessage" 
    :control="docDialogCtl"
    ref="documentMan"
  />
</template>