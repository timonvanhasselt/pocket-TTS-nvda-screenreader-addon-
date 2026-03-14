import os
import sys
import wx
import gui
import globalVars
from logHandler import log
from gui.settingsDialogs import SettingsPanel

# Locate the ONNX engine relative to this file:
# globalPlugins/pocket_tts_onnx/ -> addon root -> synthDrivers/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ADDON_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
SYNTH_DIR = os.path.join(ADDON_DIR, "synthDrivers")
LIBS_DIR = os.path.join(ADDON_DIR, "libs")

for _dir in (SYNTH_DIR, LIBS_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

try:
    import numpy as np
    from pocket_tts_onnx import PocketTTSOnnx
    log.info("Pocket TTS Settings: engine loaded.")
except ImportError as e:
    log.error(f"Pocket TTS Settings: Error loading dependencies: {e}")
    np = None
    PocketTTSOnnx = None

_ = lambda s: s


class PocketTTSSettingsPanel(SettingsPanel):
    title = _("Pocket TTS Voice Manager")

    def makeSettings(self, settingsSizer):
        conf_dir = globalVars.appArgs.configPath
        self.models_root = os.path.join(conf_dir, "pocket_tts")
        self.voices_dir = os.path.join(self.models_root, "voices")
        self.onnx_dir = os.path.join(self.models_root, "onnx")
        self.tokenizer_path = os.path.join(self.models_root, "tokenizer.model")

        os.makedirs(self.voices_dir, exist_ok=True)

        # --- Section 1: Add / Clone Voice ---
        add_box = wx.StaticBox(self, label=_("Add new voice (Voice Cloning)"))
        add_sizer = wx.StaticBoxSizer(add_box, wx.VERTICAL)

        help_text = wx.StaticText(
            self,
            label=_(
                "Select an audio file (MP3/WAV) to generate a new .npy voice embedding.\n"
                "Only the first 30 seconds of the file will be used."
            ),
        )
        add_sizer.Add(help_text, 0, wx.ALL, 5)

        self.btnAdd = wx.Button(self, label=_("&Convert audio file..."))
        self.btnAdd.Bind(wx.EVT_BUTTON, self.onAddVoice)
        add_sizer.Add(self.btnAdd, 0, wx.ALL, 5)
        settingsSizer.Add(add_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # --- Section 2: Manage Voices ---
        manage_box = wx.StaticBox(self, label=_("Manage installed voices"))
        manage_sizer = wx.StaticBoxSizer(manage_box, wx.VERTICAL)

        list_label = wx.StaticText(self, label=_("&Select a voice from the list:"))
        manage_sizer.Add(list_label, 0, wx.LEFT | wx.TOP, 5)

        self.voiceList = wx.Choice(self, choices=self._get_installed_voices())
        self.voiceList.Bind(wx.EVT_CHOICE, self.onVoiceSelect)
        manage_sizer.Add(self.voiceList, 0, wx.EXPAND | wx.ALL, 5)

        name_label = wx.StaticText(self, label=_("&Edit display name:"))
        manage_sizer.Add(name_label, 0, wx.LEFT | wx.TOP, 5)

        self.nameEdit = wx.TextCtrl(self)
        manage_sizer.Add(self.nameEdit, 0, wx.EXPAND | wx.ALL, 5)

        self.btnRename = wx.Button(self, label=_("&Rename voice file"))
        self.btnRename.Bind(wx.EVT_BUTTON, self.onRenameVoice)
        manage_sizer.Add(self.btnRename, 0, wx.ALL, 5)

        manage_sizer.AddSpacer(10)

        self.btnRemove = wx.Button(self, label=_("&Remove selected voice"))
        self.btnRemove.Bind(wx.EVT_BUTTON, self.onRemoveVoice)
        manage_sizer.Add(self.btnRemove, 0, wx.ALL, 5)

        settingsSizer.Add(manage_sizer, 0, wx.EXPAND | wx.ALL, 10)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_installed_voices(self):
        """Return list of .npy filenames in the voices directory."""
        if not os.path.exists(self.voices_dir):
            return []
        try:
            return [f for f in os.listdir(self.voices_dir) if f.lower().endswith(".npy")]
        except Exception:
            return []

    def _refresh_ui(self):
        """Reload the voice list and reset the selection."""
        voices = self._get_installed_voices()
        self.voiceList.Clear()
        self.voiceList.AppendItems(voices)
        if voices:
            self.voiceList.SetSelection(0)
            self.onVoiceSelect(None)
        else:
            self.nameEdit.Clear()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def onAddVoice(self, evt):
        """Encode an audio file and save the resulting .npy embedding."""
        if PocketTTSOnnx is None or np is None:
            gui.messageBox(
                _("Dependencies not loaded. Check the NVDA log for details."),
                _("Error"),
            )
            return

        wildcard = "Audio files (*.mp3;*.wav)|*.mp3;*.wav"
        with wx.FileDialog(
            self,
            message=_("Select a voice sample"),
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as fd:
            if fd.ShowModal() != wx.ID_OK:
                return
            audio_src = fd.GetPath()

        base_name = os.path.splitext(os.path.basename(audio_src))[0]
        dest_path = os.path.join(self.voices_dir, f"{base_name}.npy")

        gui.messageBox(
            _("Generating voice embedding. NVDA may be unresponsive for a moment…"),
            _("Processing"),
        )

        try:
            tts = PocketTTSOnnx(
                models_dir=self.onnx_dir,
                tokenizer_path=self.tokenizer_path,
                precision="int8",
                lsd_steps=1,
            )
            embedding = tts.encode_voice(audio_src)
            np.save(dest_path, embedding)
            self._refresh_ui()
            gui.messageBox(
                _("Voice '{name}' successfully created!").format(name=base_name),
                _("Success"),
            )
        except Exception as e:
            log.error(f"Pocket TTS: Conversion error: {e}")
            gui.messageBox(f"Error during conversion:\n{e}", _("Error"))

    def onRenameVoice(self, evt):
        """Rename the selected .npy file on disk."""
        old_filename = self.voiceList.GetStringSelection()
        new_name = self.nameEdit.GetValue().strip()
        if not old_filename or not new_name:
            return

        old_path = os.path.join(self.voices_dir, old_filename)
        new_path = os.path.join(self.voices_dir, f"{new_name}.npy")

        try:
            os.rename(old_path, new_path)
            self._refresh_ui()
            gui.messageBox(
                _("Voice renamed to '{name}'.").format(name=new_name),
                _("Success"),
            )
        except Exception as e:
            gui.messageBox(f"Failed to rename:\n{e}", _("Error"))

    def onRemoveVoice(self, evt):
        """Delete the selected .npy file after confirmation."""
        sel = self.voiceList.GetStringSelection()
        if not sel:
            return

        if (
            gui.messageBox(
                _("Are you sure you want to remove the voice '{name}'?").format(name=sel),
                _("Confirm"),
                wx.YES_NO | wx.ICON_QUESTION,
            )
            == wx.YES
        ):
            try:
                os.remove(os.path.join(self.voices_dir, sel))
                self._refresh_ui()
            except Exception as e:
                log.error(f"Pocket TTS: Error removing voice: {e}")

    def onVoiceSelect(self, evt):
        """Populate the name field when a voice is selected."""
        filename = self.voiceList.GetStringSelection()
        if filename:
            self.nameEdit.SetValue(os.path.splitext(filename)[0])

    def onSave(self):
        pass
