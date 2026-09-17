import React from 'react';
import { History as HistoryIcon } from 'lucide-react';
import { useApp } from '../../context/AppContext';
import { Modal, ModalHeader, EmptyState } from '../ui/ui';
import {
  diagnosisTypeMeta,
  entryTitle,
  entryConfidence,
  fullDate,
} from '../../lib/diagnosis';
import ImageResultView from '../results/ImageResultView';
import SymptomResultView from '../results/SymptomResultView';
import ScreeningResultView from '../results/ScreeningResultView';

/* Re-opens any past analysis in full. Mounted once in App. */
export default function ResultInspector() {
  const { inspectorEntry: entry, closeInspector } = useApp();
  if (!entry) return null;

  const meta = diagnosisTypeMeta(entry.type);
  const conf = entryConfidence(entry);
  const subtitle = [meta.label, entry.timestamp ? fullDate(entry.timestamp) : null, conf ? `${conf} confidence` : null]
    .filter(Boolean)
    .join('  ·  ');

  const renderBody = () => {
    if (!entry.data) {
      return (
        <EmptyState
          icon={HistoryIcon}
          title="Full detail unavailable"
          hint="This entry was saved by an older version — only the headline was kept."
        />
      );
    }
    if (entry.type?.startsWith('image_')) {
      return <ImageResultView result={entry.data} fileName={entry.fileName} />;
    }
    if (entry.type === 'symptom') {
      return <SymptomResultView result={entry.data} symptoms={entry.symptoms || []} />;
    }
    if (entry.type === 'heart' || entry.type === 'cancer') {
      return <ScreeningResultView kind={entry.type} result={entry.data} />;
    }
    return (
      <EmptyState icon={HistoryIcon} title="Unknown entry type" hint="This record can't be displayed." />
    );
  };

  return (
    <Modal onClose={closeInspector} size="xl">
      <ModalHeader title={entryTitle(entry)} subtitle={subtitle} onClose={closeInspector} />
      <div className="max-h-[74vh] overflow-y-auto">{renderBody()}</div>
    </Modal>
  );
}
